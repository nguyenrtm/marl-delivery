from env import Environment
#from agent import Agents
from DQN.DQNAgent import DQNAgents as Agents
from tqdm import tqdm
import numpy as np

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning for Delivery")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--n_packages", type=int, default=100, help="Number of packages")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--max_time_steps", type=int, default=1000, help="Maximum time steps for the environment")
    parser.add_argument("--map", type=str, default="map1.txt", help="Map name")

    args = parser.parse_args()
    np.random.seed(args.seed)

    env = Environment(map_file=args.map, max_time_steps=args.max_time_steps,
                      n_robots=args.num_agents, n_packages=args.n_packages,
                      seed = args.seed)
    
    state = env.reset()

    observation_shape = (6, env.n_rows, env.n_cols)
    agents = Agents(observation_shape, "DQN/weights/map4/map4_ep850.pt", "cpu")
    agents.init_agents(state)
    
    infos = {}
    done = False
    t = 0
    while not done:
        actions = agents.get_actions(state)
        next_state, reward, done, cur_infos = env.step(actions)
        state = next_state
        infos = cur_infos
        # env.render()
        if t % 100 == 0:
            print(t)
        t += 1

    print("Episode finished")
    print("Total reward for seed 2025:", infos['total_reward'])
    print("Total time steps:", infos['total_time_steps'])
    reward_lst = []
    for s in tqdm(range(5)):
        np.random.seed(s + 1)

        env = Environment(map_file=args.map, max_time_steps=args.max_time_steps,
                        n_robots=args.num_agents, n_packages=args.n_packages,
                        seed=s+1)

        state = env.reset()
        agents.init_agents(state)

        done = False
        t = 0
        while not done:
            actions = agents.get_actions(state)
            next_state, reward, done, infos = env.step(actions)
            state = next_state
            if t % 100 == 0:
                print(t)
            t += 1
        print(f" seed {s+1}, reward = {infos['total_reward']}")
        reward_lst.append(infos['total_reward'])
    print("Average reward seed 1 - 10 runs:", np.round(np.mean(reward_lst), 2), "+/-", np.round(np.std(reward_lst), 2))