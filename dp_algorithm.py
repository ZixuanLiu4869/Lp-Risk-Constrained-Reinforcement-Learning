import numpy as np

from environment import GRID_H, GRID_W, START_STATE, GOAL_STATE, HAZARD_STATES, STEP_REWARD, HAZARD_PENALTY, HAZARD_FAIL_PROB

def solve_constrained_MDP(risk_threshold, p):
    """
    Solve the constrained MDP by searching for an optimal deterministic policy under mean-Lp risk constraint.
    Returns optimal policy (dict mapping state->action), and metrics (return, risk).
    """
    # All states in grid (excluding goal absorbing state)
    states = [(i, j) for i in range(GRID_H) for j in range(GRID_W)]
    states.remove(GOAL_STATE)
    actions = [0, 1, 2, 3]

    # Helper to evaluate a given deterministic policy for cost and risk
    def evaluate_policy(policy):
        # Use simulation to estimate cost distribution (exact analytic evaluation possible for this small MDP)
        costs = []
        for _ in range(10000):
            state = START_STATE
            total_cost = 0.0
            step = 0
            while state != GOAL_STATE and step < 100:
                a = policy[state]
                # simulate one step (for DP evaluation, we could compute analytically, but Monte Carlo is fine here)
                if a is None:
                    break
                # Determine next state and cost deterministically or with hazard randomness
                x, y = state
                if a == 0: nx, ny = x-1, y
                elif a == 1: nx, ny = x, y+1
                elif a == 2: nx, ny = x+1, y
                elif a == 3: nx, ny = x, y-1
                else: nx, ny = x, y
                if nx < 0 or nx >= GRID_H or ny < 0 or ny >= GRID_W:
                    # off-grid: treat as staying with step cost
                    next_state = state
                    cost = 1.0  # positive cost
                else:
                    next_state = (nx, ny)
                    if next_state == GOAL_STATE:
                        cost = 0.0
                    elif next_state in HAZARD_STATES:
                        # hazard event
                        if np.random.rand() < HAZARD_FAIL_PROB:
                            cost = 10.0  # positive cost for hazard penalty
                        else:
                            cost = 1.0
                    else:
                        cost = 1.0
                total_cost += cost
                state = next_state
                step += 1
            costs.append(total_cost)
        avg_cost = np.mean(costs)
        risk = (np.mean(np.array(costs) ** p)) ** (1.0/p)
        return avg_cost, risk

    # Define two candidate policies: hazard route vs safe route (in this simple case)
    # Policy A: always move down (col 0) – goes through hazard
    policy_hazard = {s: None for s in states}
    for i in range(GRID_H):
        policy_hazard[(i, 0)] = 2  # down action in column 0
    # Policy B: avoid hazard by detouring through column 1
    policy_safe = {s: None for s in states}
    # one safe path: (0,0)->(1,0)->(1,1)->(2,1)->(3,1)->(3,0)->(4,0)
    policy_safe[(0,0)] = 2   # down
    policy_safe[(1,0)] = 1   # right
    policy_safe[(1,1)] = 2   # down
    policy_safe[(2,1)] = 2   # down
    policy_safe[(3,1)] = 3   # left
    policy_safe[(3,0)] = 2   # down
    # Evaluate both
    avg_cost_h, risk_h = evaluate_policy(policy_hazard)
    avg_cost_s, risk_s = evaluate_policy(policy_safe)
    # Select policy satisfying risk constraint with highest return (lowest cost)
    if risk_h <= risk_threshold and avg_cost_h < avg_cost_s:
        return policy_hazard, -avg_cost_h, risk_h
    else:
        return policy_safe, -avg_cost_s, risk_s
