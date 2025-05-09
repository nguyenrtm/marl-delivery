# marl-delivery
MARL samples code for Package Delivery.
You has to run and test against BFS agents for the following 5 configs.
The seeds are given at later time.
Average over 50 runs

# Testing scripts
```python main.py --seed 10 --max_time_steps 1000 --map map1.txt --num_agents 5 --n_packages 100```
greedyagent: 32.61 +/- 15.39
improved_bfs_agent: 83.65 +/- 56.35
a_star_agent: 88.39 +/- 60.32

```python main.py --seed 10 --max_time_steps 1000 --map map2.txt --num_agents 5 --n_packages 100```
greedyagent: 28.9 +/- 15.46
improved_bfs_agent: 79.05 +/- 52.93
a_star_agent: 80.96 +/- 58.65

```python main.py --seed 10 --max_time_steps 1000 --map map3.txt --num_agents 5 --n_packages 500```
greedyagent: 17.33 +/- 11.37
improved_bfs_agent: 53.57 +/- 41.0
a_star_agent: 66.95 +/- 44.58

```python main.py --seed 10 --max_time_steps 1000 --map map4.txt --num_agents 10 --n_packages 500```
greedyagent: 49.98 +/- 17.73
improved_bfs_agent: 166.56 +/- 85.54
a_star_agent: 146.33 +/- 76.91

```python main.py --seed 10 --max_time_steps 1000 --map map5.txt --num_agents 10 --n_packages 1000```
greedyagent: 15.96 +/- 13.41
improved_bfs_agent: 34.57 +/- 29.59
a_star_agent: 35.71 +/- 34.11

# For RL testing
- You can use `simple_PPO.ipynb` as the starting point.
- Avoid modify the class `Env`, you can try to modify the `convert_state` function or `reward_shaping`
- You can choose to use or change the standard `PPO`. Note that: It is not easy to match the greedy agent, using RL.


# TODO:
- [x]: Add BFS agents
- [x]: Add test scripts
- [x]: Add RL agents
