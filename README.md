# Online Multi-Agent Pickup and Delivery with Deadline Constraints
## Members
1. Nguyen Binh Nguyen (21020526)
2. Ngo Danh Lam (21021512)
3. Pham Cong Minh (22028239)

## Testing Scripts
For results on our proposed method, Floyd-Warshall Heuristic with Multi-Greedy Prioritized Planning (FW-PP), please import ```mapd-project.ipynb``` notebook on Kaggle and run.

For the A* Search with Reservation Table baseline, you can run directly using the following commands:
1. Full experiments on seed 2025:

```bash run_all.sh```

2. Full experiments over 10 random seeds:

```bash run_all_10_seeds.sh```

For the Deep Q Network baseline, you can run using the notebook at ```DQN/dqn-training.ipynb```