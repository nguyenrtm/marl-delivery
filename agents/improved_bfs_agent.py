from collections import deque
import numpy as np

# Improved BFS with distance return only
def bfs_distance(map_grid, start, goal):
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

    return float('inf')  # unreachable


def bfs_next_action(map_grid, start, goal):
    n_rows, n_cols = len(map_grid), len(map_grid[0])
    queue = deque([(start, [])])
    visited = set([start])

    while queue:
        current, path = queue.popleft()
        if current == goal and path:
            return path[0]  # Return first step toward goal

        for move, (dx, dy) in zip(['U', 'D', 'L', 'R'], [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            next_pos = (current[0] + dx, current[1] + dy)
            if 0 <= next_pos[0] < n_rows and 0 <= next_pos[1] < n_cols:
                if map_grid[next_pos[0]][next_pos[1]] == 0 and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))

    return 'S'  # Stay if unreachable


class ImprovedBFSAgents:
    def __init__(self):
        self.n_robots = 0
        self.robots = []
        self.robots_target = []
        self.packages = []
        self.packages_free = []
        self.map = None
        self.time = 0
        self.is_init = False

    def init_agents(self, state):
        self.map = state['map']
        self.n_robots = len(state['robots'])
        self.robots = [(int(r[0])-1, int(r[1])-1, int(r[2])) for r in state['robots']]
        self.robots_target = ['free'] * self.n_robots

        self.packages = [
            (int(p[0]), int(p[1])-1, int(p[2])-1, int(p[3])-1, int(p[4])-1, int(p[5]))
            for p in state['packages']
        ]
        self.packages_free = [True] * len(self.packages)
        self.time = state['time_step']

    def update_inner_state(self, state):
        self.time = state['time_step']
        for i, robot in enumerate(state['robots']):
            pos = (int(robot[0])-1, int(robot[1])-1, int(robot[2]))
            if self.robots[i][2] != 0 and pos[2] == 0:
                self.robots_target[i] = 'free'
            self.robots[i] = pos

        new_pkgs = [
            (int(p[0]), int(p[1])-1, int(p[2])-1, int(p[3])-1, int(p[4])-1, int(p[5]))
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
            pkg_id, sx, sy, tx, ty, start_time = pkg
            if self.time < start_time:
                continue

            pickup_dist = bfs_distance(self.map, robot_pos, (sx, sy))
            delivery_dist = bfs_distance(self.map, (sx, sy), (tx, ty))
            score = pickup_dist + delivery_dist
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

        actions = []
        reserved_packages = set()

        for i in range(self.n_robots):
            rx, ry, carrying = self.robots[i]

            if carrying > 0:
                pkg = self.packages[carrying - 1]
                tx, ty = pkg[3], pkg[4]
                if (rx, ry) == (tx, ty):
                    actions.append(('S', '2'))
                else:
                    move = bfs_next_action(self.map, (rx, ry), (tx, ty))
                    actions.append((move, '0'))
            else:
                if self.robots_target[i] == 'free':
                    pkg_idx = self.find_best_package((rx, ry))
                    if pkg_idx is not None:
                        self.robots_target[i] = self.packages[pkg_idx][0]
                        self.packages_free[pkg_idx] = False
                        reserved_packages.add(pkg_idx)
                    else:
                        actions.append(('S', '0'))
                        continue

                target_pkg_id = self.robots_target[i]
                pkg = self.packages[target_pkg_id - 1]
                sx, sy = pkg[1], pkg[2]

                if (rx, ry) == (sx, sy):
                    actions.append(('S', '1'))
                else:
                    move = bfs_next_action(self.map, (rx, ry), (sx, sy))
                    actions.append((move, '0'))

        return actions
