from env_with_render import Environment
# from agents.a_star_aware_agent import AStarAwareAgents as Agents
# from agents.greedyagent import GreedyAgents as Agents
from astarcbs import AStarAgents as Agents
import pygame
import numpy as np

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
    np.random.seed(args.seed)

    env = Environment(map_file=args.map, max_time_steps=args.max_time_steps,
                      n_robots=args.num_agents, n_packages=args.n_packages,
                      seed = args.seed)
    
    state = env.reset()
    agents = Agents()
    agents.init_agents(state)
    print(state)
    env.render()
    done = False
    t = 0
    infos = {}
    

    while not done:
        actions = agents.get_actions(state)
        next_state, reward, new_done, new_infos = env.step(actions)
        state = next_state
        # env.render()
        infos = new_infos
        if new_done:
            done = True
        if not done:
            try:
                env.render_animate()
                pygame.time.wait(0)
            except pygame.error as e:
                print(f"Pygame error: {e}")
                done = True
                if not new_done:
                    infos['total_reward'] = env.total_reward
                    infos['toral_time_steps'] = env.t
        if done:
            break
        
        t += 1

    print("Episode finished")
    print("Total reward:", infos['total_reward'])
    print("Total time steps:", infos['total_time_steps'])