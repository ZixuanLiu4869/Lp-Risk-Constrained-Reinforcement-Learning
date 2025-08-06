# Mean-Lp Risk-Constrained RL Experiment Suite

This repository contains an implementation of two algorithms evaluated on a 5x5 GridWorld for **mean-$L_p$ risk-constrained reinforcement learning**:
1. **Primal-Dual Policy Gradient** (`pd_algorithm.py`): Model-free policy gradient algorithm with Lagrangian relaxation for the risk constraint.
2. **Dynamic Programming with Augmented State** (`dp_algorithm.py`): Model-based approach that solves the MDP with an augmented state or by explicit enumeration to satisfy the risk constraint.

## Environment
The environment (`environment.py`) is a $5\\times5$ grid with:
- A **start state** at the top-left (S = (0,0)).
- A **goal state** at the bottom-left (G = (4,0)), which terminates the episode (reward 0 upon reaching).
- A **hazardous state** H = (2,0) with a step penalty of -10 (cost 10). Entering H incurs a penalty with probability 0.1 (otherwise only the normal step cost).
- All other transitions have a step cost of -1 per move. The agent can move up, down, left, right (stochastic outcomes for hazard events only).

## Usage
- Run `train.py` to train the policy using the primal-dual method for $p = 1, 2, 4, 8$. Training logs average return and risk (constraint satisfaction) per iteration. It also computes the optimal policy via the DP method for comparison.
- After training, use `evaluate.py` to compute final performance metrics of the learned policy.
- Use `plots.py` to generate:
  - **Policy heatmaps** showing the learned policy for each $p$ (arrows indicate actions in each state; H and G are hazard and goal) and 
  - **Risk-Return curves** comparing final average return and risk (mean-$L_p$ cost).
 

## Extending the Code
- The risk probability and penalty can be adjusted in `environment.py` to simulate different hazard severities.
- Additional risk measures (e.g., CVaR) can be implemented by modifying the constraint term in the policy gradient (e.g., replacing $C^p$ with an indicator for CVaR threshold exceedance).
- For larger or more complex environments, consider function approximation for the policy (e.g., neural network) and use `PrimalDualPG` as a template for implementing constrained policy optimization.


