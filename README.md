# marl-delivery
MARL samples code for Package Delivery.
You has to run and test against BFS agents for the following 5 configs.
The seeds are given at later time.

- Version v1.1: Fix a small logic of `start_time` should less than `dead_line`

# Testing scripts
Results averaged over 10 runs
```python main.py --seed 10 --max_time_steps 1000 --map map1.txt --num_agents 5 --n_packages 100```

greedyagent: 42.45 +/- 14.04

improved_bfs_agent: 80.81 +/- 36.0

a_star_agent: 120.23 +/- 72.91

a_star_aware_agent: 901.4 +/- 36.31

```python main.py --seed 10 --max_time_steps 1000 --map map2.txt --num_agents 5 --n_packages 100```

greedyagent: 26.4 +/- 17.24

improved_bfs_agent: 62.28 +/- 29.83

a_star_agent: 115.37 +/- 86.89

a_star_aware_agent: 725.12 +/- 85.13

```python main.py --seed 10 --max_time_steps 1000 --map map3.txt --num_agents 5 --n_packages 500```

greedyagent: 14.4 +/- 11.07

improved_bfs_agent: 48.49 +/- 29.35

a_star_agent: 59.0 +/- 31.91

a_star_aware_agent: 707.37 +/- 274.07

```python main.py --seed 10 --max_time_steps 1000 --map map4.txt --num_agents 10 --n_packages 500```

greedyagent: 44.99 +/- 13.5

improved_bfs_agent: 142.23 +/- 53.87

a_star_agent: 160.24 +/- 71.89

a_star_aware_agent: 2841.96 +/- 90.61

```python main.py --seed 10 --max_time_steps 1000 --map map5.txt --num_agents 10 --n_packages 1000```

greedyagent: 20.0 +/- 15.04

improved_bfs_agent: 37.4 +/- 29.72

a_star_agent: 59.53 +/- 36.09

a_star_aware_agent: 141.35 +/- 82.57