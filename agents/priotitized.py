import networkx as nx
import heapq
import math
import copy
import random
class IRobot:
    def __init__(self, position):
        self.position = position
        self.carrying = 0
        self.target = None
        self.path = []
        self.state = 0

    def update(self, position, carrying):
        self.position = position
        self.carrying = carrying
class Agents:

    def __init__(self):
        self.agents = []
        self.packages = []
        self.packages_free = []
        self.n_robots = 0
        self.state = None
        self.is_init = False
        self.heuristic = None
        self.time = 0

    def init_agents(self, state):
        self.state = state
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.robots = [IRobot((r[0]-1, r[1]-1)) for r in state['robots']]
        self.packages += [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5],p[6]) for p in state['packages']]
        self.packages_free = [True] * len(self.packages)
        self.build_heuristic()

    def get_actions(self, state):
        if self.is_init == False:
            # This mean we have invoke the init agents, use the update_inner_state to update the state
            self.is_init = True
            # self.update_inner_state(state)

        else:
            self.update_inner_state(state)

        move_actions = []
        package_actions = [0] * self.n_robots
        self.greedy()
        goals = []
        for i in range(self.n_robots):
            if self.robots[i].target is not None:
                if self.robots[i].state == 1:
                    package = self.packages[self.robots[i].target]
                    goals.append((package[1], package[2]))
                elif self.robots[i].state == 2:
                    package = self.packages[self.robots[i].target]
                    goals.append((package[3], package[4]))
            else:
                goals.append((self.robots[i].position[0], self.robots[i].position[1]))
        plans = self.prioritized_planning(goals)
        for i in range(self.n_robots):
            if not plans[i]:
                move_actions.append("S")
            else:
                delta = (plans[i][1][0] - plans[i][0][0], plans[i][1][1] - plans[i][0][1])
                if delta == (0, 1):
                    move_actions.append("R")
                elif delta == (1, 0):
                    move_actions.append("D")
                elif delta == (0, -1):
                    move_actions.append("L")
                elif delta == (-1, 0):
                    move_actions.append("U")
                else:
                    move_actions.append("S")       
        for i in range(self.n_robots):
            plan = plans[i]
            if not plan:
                if self.robots[i].state == 1:
                    package = self.packages[self.robots[i].target]
                    if self.robots[i].position == (package[1], package[2]):
                        self.robots[i].state = 2
                        self.robots[i].carrying = package[0]
                        package_actions[i] = 1
                elif self.robots[i].state == 2:
                    package = self.packages[self.robots[i].target]
                    if self.robots[i].position == (package[3], package[4]):
                        self.robots[i].state = 0
                        self.robots[i].carrying = 0
                        package_actions[i] = 2
                continue
            if self.robots[i].state == 1:  # Robot is assigned to pick up a package
                package = self.packages[self.robots[i].target]
                if plan[1] == (package[1], package[2]):
                    self.robots[i].state = 2  # Update state to carrying the package
                    self.robots[i].carrying = package[0]
                    package_actions[i] = 1
            if self.robots[i].state == 2:
                package = self.packages[self.robots[i].target]
                if plan[1] == (package[3], package[4]):
                    self.robots[i].state = 0
                    self.robots[i].carrying = 0
                    package_actions[i] = 2
        
        actions = []
        for i in range(self.n_robots):
            actions.append((move_actions[i], str(package_actions[i])))
        return actions
          
    
    def is_valid(self, position):
        x, y = position
        if x < 0 or x >= len(self.map) or y < 0 or y >= len(self.map[0]):
            return False
        if self.map[x][y] == 1:
            return False
        return True
    
    def build_heuristic(self):
        self.nx_grid = nx.grid_2d_graph(len(self.map), len(self.map[0]))
        for x in range(len(self.map)):
            for y in range(len(self.map[0])):
                if self.map[x][y] == 1:
                    self.nx_grid.remove_node((x, y))
        self.heuristic = dict(nx.floyd_warshall(self.nx_grid))
        
    def update_inner_state(self,state):
        # Update robot positions and states
        for i in range(len(state['robots'])):
            robot = state['robots'][i]
            self.robots[i].update((robot[0]-1, robot[1]-1), robot[2])
            
        # Update package positions and states
        self.packages += [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5],p[6]) for p in state['packages']]
        self.packages_free += [True] * len(state['packages'])
        self.time = state['time_step']
        
    def space_time_astar(self, start, goal,constraints=None, goal_constraints=None):
        """
        A* algorithm to find the shortest path in a space-time graph using heapq
        
        """
        open_list = []
        closed_set = set()
        start_node = A_Star_Node(start, 0)
        goal_node = A_Star_Node(goal, 0)
        start_node.g = 0
        start_node.h = self.heuristic[start][goal]
        start_node.f = start_node.g + start_node.h
        heapq.heappush(open_list, start_node)
        while open_list:
            current_node = heapq.heappop(open_list)
            if current_node.position == goal_node.position:
                return current_node.get_path()
            closed_set.add(current_node)
            #max size of closed set, terminate if too large
            if len(closed_set) > 500:
                return None
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_position = (current_node.position[0] + dx, current_node.position[1] + dy)
                if new_position in closed_set or not self.is_valid(new_position) or (constraints and self.is_constrained(current_node.position, new_position, current_node.time + 1, constraints,goal_constraints)):
                    continue
                new_time = current_node.time + 1
                new_node = A_Star_Node(new_position, new_time, current_node)
                if new_node in closed_set:
                    continue
                new_node.g = current_node.g + 1
                new_node.h = self.heuristic[new_position][goal]
                new_node.f = new_node.g + new_node.h
                new_node.move = current_node.move + 1
                if new_node not in open_list :
                    heapq.heappush(open_list, new_node)
                if new_node in open_list:
                    for node in open_list:
                        if node == new_node and node.f > new_node.f:
                            open_list.remove(node)
                            heapq.heappush(open_list, new_node)
                            break
            #stand still
            if (constraints and self.is_constrained(current_node.position, current_node.position, current_node.time + 1, constraints,goal_constraints)):
                continue
            new_node = A_Star_Node(current_node.position, current_node.time + 1, current_node)
            new_node.g = current_node.g + 1
            new_node.h = current_node.h
            new_node.f = current_node.f + 1
            new_node.move = current_node.move
            if new_node not in closed_set and new_node not in open_list:
                heapq.heappush(open_list, new_node)
        return None
    
    def prioritized_planning(self, goals):
        constraints = set()
        goal_constraints = dict()
        plan = [None] * self.n_robots
        for i in range(self.n_robots):
            path = []
            if (self.robots[i].position) != goals[i]:
                path = self.space_time_astar((self.robots[i].position), goals[i], constraints,goal_constraints)
                if path is None:
                    plan[i] = None
                    goal_constraints[self.robots[i].position] = 0
                    continue
                for j, vertex in enumerate(path):
                    constraints.add((vertex[0], vertex[1], j))
                #goal constraints
                goal_constraints[goals[i]] = len(path)
                # edge constraints
                for k in range(len(path) - 1):
                    constraints.add((path[k][0], path[k][1], path[k + 1][0], path[k + 1][1], k+1))
            plan[i] = path
        return plan
    
    def is_constrained(self, current_position, planed_position, time, constraints, goal_constraints=None):
        if (planed_position[0], planed_position[1], time) in constraints:
            return True
        if (current_position[0], current_position[1], planed_position[0], planed_position[1], time) in constraints:
            return True
        if (planed_position[0], planed_position[1], current_position[0], current_position[1], time) in constraints:
            return True
        if (planed_position) in goal_constraints:
            if time >= goal_constraints[planed_position]:
                return True
        #circular constraints, bfs check for cycles
        return False
    
    def greedy(self):
        #greedy assign packages to robots
        order = self.greedy_random()
        for i in order:
            if self.robots[i].state == 0:
                #robot is free
                min_dist = float('inf')
                min_pkg = None
                for j in range(len(self.packages)):
                    if self.packages_free[j]:
                        dist = self.heuristic[self.robots[i].position][self.packages[j][1], self.packages[j][2]]
                        delivering_dist = self.heuristic[self.packages[j][3], self.packages[j][4]][self.packages[j][1], self.packages[j][2]]
                        total_dist = dist + delivering_dist
                        if self.time + total_dist > self.packages[j][6]:
                            continue
                        if dist < min_dist:
                            min_dist = dist
                            min_pkg = j
                        if dist == min_dist:
                            if self.packages[j][0] < self.packages[min_pkg][0]:
                                min_pkg = j
                self.robots[i].target = min_pkg
                if min_pkg is not None:
                    self.robots[i].state = 1
                    self.packages_free[min_pkg] = False
        # for i in range(self.n_robots):
        #     if self.robots[i].state == 0: #robot is free
        #         #robot is free
        #         min_dist = float('inf')
        #         min_pkg = None
        #         for j in range(len(self.packages)):
        #             if self.packages_free[j]:
        #                 dist = self.heuristic[self.robots[i].position][self.packages[j][1], self.packages[j][2]]
        #                 delivering_dist = self.heuristic[self.packages[j][3], self.packages[j][4]][self.packages[j][1], self.packages[j][2]]
        #                 total_dist = dist + delivering_dist
        #                 if self.time + total_dist > self.packages[j][6]:
        #                     continue
        #                 if dist < min_dist:
        #                     min_dist = dist
        #                     min_pkg = j
        #                 if dist == min_dist:
        #                     if self.packages[j][0] < self.packages[min_pkg][0]:
        #                         min_pkg = j
        #         self.robots[i].target = min_pkg
        #         if min_pkg is not None:
        #             self.robots[i].state = 1
        #             self.packages_free[min_pkg] = False
                
            

    
    def greedy_random(self):
        #get free robots indexes
        free_robots = []
        for i in range(self.n_robots):
            if self.robots[i].state == 0:
                free_robots.append(i)
        number_of_random = min(10,math.factorial(len(free_robots)))
        min_order = []
        min_value = float('inf')
        order_set= set()
        for i in range(number_of_random):
            free_robots = random.shuffle(free_robots)
            while (free_robots[0], free_robots[1]) in order_set:
                free_robots = self.generate_random_shuffle(free_robots)
            order_set.add((free_robots[0], free_robots[1]))
            value = 0
            temp_packages = copy.deepcopy(self.packages)
            temp_packages_free = copy.deepcopy(self.packages_free)
            for i in range(len(free_robots)):
                robot = free_robots[i]
                if self.robots[robot].state == 0: #robot is free
                    #robot is free
                    min_dist = float('inf')
                    min_pkg = None
                    for j in range(len(temp_packages)):
                        if temp_packages_free[j]:
                            dist = self.heuristic[self.robots[robot].position][temp_packages[j][1], temp_packages[j][2]]
                            # delivering_dist = self.heuristic[temp_packages[j][3], temp_packages[j][4]][temp_packages[j][1], temp_packages[j][2]]
                            # total_dist = dist + delivering_dist
                            # if self.time + total_dist > temp_packages[j][6]:
                            #     continue
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
        
        
    

class A_Star_Node:
    def __init__(self, position, time, parent=None):
        self.position = position
        self.time = time
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0
        self.move = 0
        
    def __lt__(self, other):
        if self.f == other.f:
            return self.move < other.move
        return self.f < other.f
    
    def __eq__(self, value):
        return self.position == value.position and self.time == value.time
    
    def __hash__(self):
        return hash((self.position, self.time))
    
    def get_path(self):
        path = []
        current = self
        while current:
            path.append(current.position)
            current = current.parent
        return path[::-1]
    
    
if __name__ == "__main__":
    agents = Agents()
    state = {
        'robots': [(3, 1), (3, 5)],
        'packages': [],
        'map': [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ]
    }
    agents.init_agents(state)
    plan = agents.prioritized_planning([(1,3), (5, 3)])
    #visualize the path
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(state['map'][0]))
    ax.set_ylim(0, len(state['map']))
    ax.set_aspect('equal')
    ax.invert_yaxis()
    for x in range(len(state['map'])):
        for y in range(len(state['map'][0])):
            if state['map'][x][y] == 1:
                ax.add_patch(patches.Rectangle((y, x), 1, 1, color='black'))
            else:
                ax.add_patch(patches.Rectangle((y, x), 1, 1, color='white'))
    # for robot in agents.robots:
    #     ax.add_patch(patches.Rectangle((robot[1], robot[0]), 1, 1, color='blue'))
    # for pkg in agents.packages:
    #     ax.add_patch(patches.Rectangle((pkg[1], pkg[0]), 1, 1, color='red'))
    print(plan)
   