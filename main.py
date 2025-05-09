from env import Environment
from greedyagent import GreedyAgents as Agents
# from improved_bfs_agent import ImprovedBFSAgents as Agents
# from a_star_agent import AStarAgents as Agents

import numpy as np
import os
import sys
from datetime import datetime

os.makedirs("logs", exist_ok=True)
log_filename = f"logs/run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_filename, "w")

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.__stdout__, log_file)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning for Delivery")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--n_packages", type=int, default=10, help="Number of packages")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--max_time_steps", type=int, default=1000, help="Maximum time steps for the environment")
    parser.add_argument("--map", type=str, default="map.txt", help="Map name")

    args = parser.parse_args()

    reward_lst = []

    for s in range(50):
        np.random.seed(s)

        env = Environment(map_file=args.map, max_time_steps=args.max_time_steps,
                        n_robots=args.num_agents, n_packages=args.n_packages,
                        seed=s)
        
        state = env.reset()
        agents = Agents()
        agents.init_agents(state)
        # print(state)
        #env.render()
        done = False
        t = 0
        while not done:
            actions = agents.get_actions(state)
            next_state, reward, done, infos = env.step(actions)
            state = next_state
            # print("Step:", t)
            # env.render()
            t += 1
        print("\nReward:", infos['total_reward'])
        print('=========Episode finished=========')
        reward_lst.append(infos['total_reward'])
    print("Average reward over 50 runs:", np.round(np.mean(reward_lst), 2), "+/-", np.round(np.std(reward_lst), 2))