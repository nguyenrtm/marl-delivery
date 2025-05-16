import copy
import math
import networkx as nx
import heapq
import random

def generate_random_shuffle(list):
        random.shuffle(list)
        return list

# Manhattan distance heuristic
def manhattan(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

# A* with Manhattan heuristic, returns only the distance
def astar_distance(map_grid, start, goal):
    n_rows, n_cols = len(map_grid), len(map_grid[0])
    open_set = [(manhattan(start, goal), 0, start)]
    visited = set()

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            return cost
        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            next_pos = (nx, ny)
            if 0 <= nx < n_rows and 0 <= ny < n_cols:
                if map_grid[nx][ny] == 0 and next_pos not in visited:
                    new_cost = cost + 1
                    est_total = new_cost + manhattan(next_pos, goal)
                    heapq.heappush(open_set, (est_total, new_cost, next_pos))

    return float('inf')

# A* with Manhattan heuristic, returns first move toward goal
def astar_next_action(map_grid, start, goal, occupied_positions=None, reservation_table=None, current_time=0, robot_id=None):
    n_rows, n_cols = len(map_grid), len(map_grid[0])
    open_set = [(manhattan(start, goal), 0, start, [])]
    visited = set()

    if occupied_positions is None:
        occupied_positions = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current == goal and path:
            return path[0]
        if current in visited:
            continue
        visited.add(current)

        for move, (dx, dy) in zip(['U', 'D', 'L', 'R'], [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            nx, ny = current[0] + dx, current[1] + dy
            next_pos = (nx, ny)
            if 0 <= nx < n_rows and 0 <= ny < n_cols:
                if map_grid[nx][ny] == 0 and next_pos not in visited:
                    if reservation_table:
                        for t in range(1, 4):  # check next 3 time steps
                            if (nx, ny, cost + current_time + t) in reservation_table:
                                break
                        else:
                            pass  # safe to proceed
                            new_cost = cost + 1
                            est_total = new_cost + manhattan(next_pos, goal)
                            if next_pos in occupied_positions:
                                est_total += n_rows / 2  # discourage paths through occupied cells
                            heapq.heappush(open_set, (est_total, new_cost, next_pos, path + [move]))
                        continue  # cell is reserved, skip
                    else:
                        new_cost = cost + 1
                        est_total = new_cost + manhattan(next_pos, goal)
                        if next_pos in occupied_positions:
                            est_total += n_rows / 2  # discourage paths through occupied cells
                        heapq.heappush(open_set, (est_total, new_cost, next_pos, path + [move]))

    return 'S'  # Stay if unreachable

class AStarAwareAgents:
    def __init__(self):
        self.n_robots = 0
        self.robots = []
        self.robots_target = []
        self.packages = []
        self.packages_free = []
        self.map = None
        self.time = 0
        self.is_init = False
        self.reservation_table = {}

    def init_agents(self, state):
        self.map = state['map']
        self.n_robots = len(state['robots'])
        self.robots = [(int(r[0]) - 1, int(r[1]) - 1, int(r[2])) for r in state['robots']]
        self.robots_target = ['free'] * self.n_robots

        self.packages = [
            (int(p[0]), int(p[1]) - 1, int(p[2]) - 1, int(p[3]) - 1, int(p[4]) - 1, int(p[5]), int(p[6]))
            for p in state['packages']
        ]
        self.packages_free = [True] * len(self.packages)
        self.time = state['time_step']
        self.build_heuristic()
        
    def build_heuristic(self):
        self.nx_grid = nx.grid_2d_graph(len(self.map), len(self.map[0]))
        for x in range(len(self.map)):
            for y in range(len(self.map[0])):
                if self.map[x][y] == 1:
                    self.nx_grid.remove_node((x, y))
        self.heuristic = dict(nx.floyd_warshall(self.nx_grid))    

    def update_inner_state(self, state):
        self.time = state['time_step']
        for i, robot in enumerate(state['robots']):
            pos = (int(robot[0]) - 1, int(robot[1]) - 1, int(robot[2]))
            if self.robots[i][2] != 0 and pos[2] == 0:
                self.robots_target[i] = 'free'
            self.robots[i] = pos

        new_pkgs = [
            (int(p[0]), int(p[1]) - 1, int(p[2]) - 1, int(p[3]) - 1, int(p[4]) - 1, int(p[5]), int(p[6]))
            for p in state['packages']
        ]
        self.packages.extend(new_pkgs)
        self.packages_free.extend([True] * len(new_pkgs))

    def find_best_package(self, robot_pos):
        best_score = float('inf')
        best_idx = None

        for idx, pkg in enumerate(self.packages):
            if not self.packages_free[idx]:
                continue

            pkg_id, sx, sy, tx, ty, start_time, deadline = pkg
            if self.time < start_time:
                continue

            pickup_dist = astar_distance(self.map, robot_pos, (sx, sy))
            # delivery_dist = astar_distance(self.map, (sx, sy), (tx, ty))
            total_travel = pickup_dist

            # Estimated arrival time at delivery location
            arrival_time = self.time + total_travel

            # Skip packages that cannot possibly be delivered in time
            if arrival_time > deadline:
                 continue

            score = pickup_dist

            if score < best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def get_actions(self, state):
        if not self.is_init:
            self.is_init = True
            self.init_agents(state)
        else:
            self.update_inner_state(state)

        actions = [None] * self.n_robots
        reserved_packages = set()

        occupied_positions = {(r[0], r[1]) for r in self.robots}
        self.reservation_table = {(r[0], r[1], self.time): i for i, r in enumerate(self.robots)}

        robot_indices = list(range(self.n_robots))

        for i in robot_indices:
            rx, ry, carrying = self.robots[i]

            if carrying > 0:
                pkg = self.packages[carrying - 1]
                tx, ty = pkg[3], pkg[4]
                if (rx, ry) == (tx, ty):
                    move = 'S'
                    action = '2'
                    actions[i] = (move, action)
                else:
                    move = astar_next_action(self.map, (rx, ry), (tx, ty), occupied_positions, self.reservation_table, self.time, i)
                    action = '0'
                    dxdy = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
                    if move in dxdy:
                        dx, dy = dxdy[move]
                        next_pos = (rx + dx, ry + dy)
                    actions[i] = (move, action)
                    if move != 'S':
                        dxdy2 = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
                        if move in dxdy2:
                            dx, dy = dxdy2[move]
                            next_pos = (rx + dx, ry + dy)
                            for t in range(1, 4):
                                self.reservation_table[(next_pos[0], next_pos[1], self.time + t)] = i
        order = self.greedy_random()
        for i in order:
            rx, ry, carrying = self.robots[i]
            if self.robots_target[i] == 'free':
                pkg_idx = self.find_best_package((rx, ry))
                if pkg_idx is not None:
                    self.robots_target[i] = self.packages[pkg_idx][0]
                    self.packages_free[pkg_idx] = False
                    reserved_packages.add(pkg_idx)
                else:
                    actions[i] = ('S', '0')
                    continue

            target_pkg_id = self.robots_target[i]
            pkg = self.packages[target_pkg_id - 1]
            sx, sy = pkg[1], pkg[2]

            if (rx, ry) == (sx, sy):
                move = 'S'
                action = '1'
                actions[i] = (move, action)
            else:
                move = astar_next_action(self.map, (rx, ry), (sx, sy), occupied_positions, self.reservation_table, self.time, i)
                action = '0'
                dxdy = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
                if move in dxdy:
                    dx, dy = dxdy[move]
                    next_pos = (rx + dx, ry + dy)
                actions[i] = (move, action)
                if move != 'S':
                    dxdy2 = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
                    if move in dxdy2:
                        dx, dy = dxdy2[move]
                        next_pos = (rx + dx, ry + dy)
                        for t in range(1, 4):
                            self.reservation_table[(next_pos[0], next_pos[1], self.time + t)] = i            
        return actions
    
    def greedy_random(self):
        #get free robots indexes
        free_robots = []
        for i in range(self.n_robots):
            if self.robots[i][2] == 0:
                free_robots.append(i)
        number_of_random = min(200,math.factorial(len(free_robots)))
        min_order = []
        min_value = float('inf')
        set = {}
        for i in range(number_of_random):
            free_robots = generate_random_shuffle(free_robots)
            while str(free_robots) in set:
                free_robots = generate_random_shuffle(free_robots)
            value = 0
            temp_packages = copy.deepcopy(self.packages)
            temp_packages_free = copy.deepcopy(self.packages_free)
            for i in range(len(free_robots)):
                robot = free_robots[i]
                min_dist = float('inf')
                min_pkg = None
                for j in range(len(temp_packages)):
                    if temp_packages_free[j]:
                        dist = self.heuristic[(self.robots[robot][0], self.robots[robot][1])][(temp_packages[j][1], temp_packages[j][2])]
                        if dist < min_dist:
                            min_dist = dist
                            min_pkg = j
                        if dist == min_dist:
                            if temp_packages[j][0] < temp_packages[min_pkg][0]:
                                min_pkg = j
                    if min_pkg is not None:
                        temp_packages_free[min_pkg] = False
                        value += min_dist
            if value < min_value:
                min_value = value
                min_order = free_robots
        return min_order