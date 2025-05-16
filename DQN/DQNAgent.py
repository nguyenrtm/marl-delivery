import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from collections import deque

NUM_MOVE_ACTIONS = 5  # S, L, R, U, D
NUM_PKG_OPS    = 3  # None, Pickup, Drop
ACTION_DIM = NUM_MOVE_ACTIONS  * NUM_PKG_OPS


class AgentNetwork(nn.Module):
    def __init__(self, observation_shape, action_dim):
        super(AgentNetwork, self).__init__()
        # observation_shape is (C, H, W)
        self.conv1 = nn.Conv2d(observation_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        flat_size = 32 * observation_shape[1] * observation_shape[2]  # 32 * H * W

        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, obs):
        # obs: (N, C, H, W) or (C, H, W)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)  # (1, C, H, W)
        x = F.relu(self.bn1(self.conv1(obs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten(start_dim=1)  # (N, 32*H*W)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

def bfs_distance(map_grid, start_x, start_y, goal_x, goal_y): #0-indexed
    start = (start_x, start_y)
    goal = (goal_x, goal_y)
    
    n_rows = len(map_grid)
    n_cols = len(map_grid[0])
    queue = deque([(goal, 0)])
    visited = set([goal])

    while queue:
        current, dist = queue.popleft()
        if current == start:
            return dist

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (current[0] + dx, current[1] + dy)
            if 0 <= next_pos[0] < n_rows and 0 <= next_pos[1] < n_cols:
                if map_grid[next_pos[0]][next_pos[1]] == 0 and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, dist + 1))

    return float('inf')  

def convert_state(state, persistent_packages, current_robot_idx):
    grid = np.array(state["map"])
    n_rows, n_cols = grid.shape
    n_channels = 6
    tensor = np.zeros((n_channels, n_rows, n_cols), dtype=np.float32)

    #Channel 0: Map
    tensor[0] = grid

    current_time_step = state["time_step"]
    if isinstance(current_time_step, np.ndarray): 
        current_time_step = current_time_step[0]

    if current_robot_idx < 0 or current_robot_idx >= len(state["robots"]):
        print("Invalid robot idx")
        return tensor 

    #current robot data
    current_robot_data = state["robots"][current_robot_idx]
    carried_pkg_id_by_current_robot = current_robot_data[2] 

    #Channel 1: Urgency of 'waiting' packages (if not carrying) 
    #Channel 2: Start positions of 'waiting' packages (if not carrying)
    if carried_pkg_id_by_current_robot == 0: # Robot is not carrying
        for pkg_id, pkg_data in persistent_packages.items():
            if pkg_data['status'] == 'waiting':
                # all 1-indexed
                rr, rc, _ = current_robot_data
                dr, dc = pkg_data['target_pos']
                sr, sc = pkg_data['start_pos'] 
                st = pkg_data['start_time']
                dl = pkg_data['deadline']

                travel_length = bfs_distance(grid, rr - 1, rc - 1, sr - 1, sc - 1) + bfs_distance(grid, sr - 1, sc - 1, dr - 1, dc - 1)
                time_left = dl - current_time_step - travel_length
                #check active package
                if current_time_step >= st:
                    urgency = 0
                    # If no time left then no urgency
                    if dl > st and time_left > 0:
                        urgency = 1.0/time_left
      

                    if 0 <= sr < n_rows and 0 <= sc < n_cols: 
                        tensor[1, sr, sc] = max(tensor[1, sr, sc], urgency) 

                    # Channel 2: Start position
                    if 0 <= sr < n_rows and 0 <= sc < n_cols: 
                        tensor[2, sr, sc] = 1.0 
    

    # Channel 3: Other robots' positions
    for i, rob_data in enumerate(state["robots"]):
        if i == current_robot_idx:
            continue 
        rr, rc, _ = rob_data 
        rr_idx, rc_idx = int(rr) - 1, int(rc) - 1 
        if 0 <= rr_idx < n_rows and 0 <= rc_idx < n_cols: 
            tensor[3, rr_idx, rc_idx] = 1.0

    #Channel 4: Current robot's position
    crr, crc, _ = current_robot_data 
    crr_idx, crc_idx = int(crr) - 1, int(crc) - 1 
    if 0 <= crr_idx < n_rows and 0 <= crc_idx < n_cols: 
        tensor[4, crr_idx, crc_idx] = 1.0

    #Channel 5: Current robot's carried package target (if robot is carrying)
    if carried_pkg_id_by_current_robot != 0:
        if carried_pkg_id_by_current_robot in persistent_packages:
            pkg_data_carried = persistent_packages[carried_pkg_id_by_current_robot]
            tr_carried, tc_carried = pkg_data_carried['target_pos'] 
            if 0 <= tr_carried < n_rows and 0 <= tc_carried < n_cols: 
                tensor[5, tr_carried, tc_carried] = 1.0

    return tensor


class DQNAgents:
    def __init__(self, observation_shape, weights_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.le_move = LabelEncoder().fit(['S','L','R','U','D'])
        self.le_pkg  = LabelEncoder().fit(['0','1','2']) #
        self.model = AgentNetwork(observation_shape, ACTION_DIM).to(self.device) 
        if weights_path is not None:
            try:
                print(f"Loading model weights from: {weights_path}")
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                print("Model weights loaded successfully.")
            except Exception as e:
                print(f"Error loading model weights: {e}")

        self.model.eval()
        self.n_robots = 0
        self.is_init = False
        self.persistent_packages = {} # Initialize persistent_packages

    def _update_persistent_packages(self, current_env_state):
        if 'packages' in current_env_state and current_env_state['packages'] is not None:
            for pkg_tuple in current_env_state['packages']:
                pkg_id = pkg_tuple[0]
                if pkg_id not in self.persistent_packages:
                    self.persistent_packages[pkg_id] = {
                        'id': pkg_id,
                        'start_pos': (pkg_tuple[1] - 1, pkg_tuple[2] - 1), 
                        'target_pos': (pkg_tuple[3] - 1, pkg_tuple[4] - 1), 
                        'start_time': pkg_tuple[5],
                        'deadline': pkg_tuple[6],
                        'status': 'waiting'
                    }

        current_carried_pkg_ids_set = set()
        if 'robots' in current_env_state and current_env_state['robots'] is not None:
            for r_idx, r_data in enumerate(current_env_state['robots']):
                carried_id = r_data[2]
                if carried_id != 0: 
                    current_carried_pkg_ids_set.add(carried_id)

        packages_to_remove_definitively = []

        for pkg_id, pkg_data in list(self.persistent_packages.items()): #
            original_status_in_tracker = pkg_data['status']

            if pkg_id in current_carried_pkg_ids_set:
                self.persistent_packages[pkg_id]['status'] = 'in_transit'
            else:
                if original_status_in_tracker == 'in_transit':
                    packages_to_remove_definitively.append(pkg_id)

        for pkg_id_to_remove in packages_to_remove_definitively:
            if pkg_id_to_remove in self.persistent_packages:
                del self.persistent_packages[pkg_id_to_remove]

    def init_agents(self, state):
        self.n_robots = len(state.get('robots', []))
        self._update_persistent_packages(state) 
        self.is_init = True

    def get_actions(self, state):

        self._update_persistent_packages(state)
        
        actions = []
        for i in range(self.n_robots):
            if 0 < 0.: 
                joint_idx = np.random.randint(0, ACTION_DIM)
            else:
    
                obs = convert_state(state, self.persistent_packages, current_robot_idx=i)
                obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
                if obs_t.dim() == 3: 
                    obs_t = obs_t.unsqueeze(0)
                
                with torch.no_grad():
                    logits = self.model(obs_t)  
                    joint_idx = logits.argmax(dim=1).item()
            
            move_idx   = joint_idx % NUM_MOVE_ACTIONS
            pkg_idx    = joint_idx // NUM_MOVE_ACTIONS
            
            move_str   = self.le_move.inverse_transform([move_idx])[0]
            pkg_str    = self.le_pkg.inverse_transform([pkg_idx])[0]
            actions.append((move_str, pkg_str))
        return actions

