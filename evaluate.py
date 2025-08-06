import numpy as np
from pd_algorithm import PrimalDualPG

def evaluate_policy(agent, p, episodes=1000):
    """Evaluate a trained agent's average return and mean-Lp risk."""
    costs = []
    for _ in range(episodes):
        _, total_cost = agent.run_episode()
        costs.append(total_cost)
    avg_return = -np.mean(costs)
    risk = (np.mean(np.array(costs) ** p)) ** (1.0/p)
    return avg_return, risk

# Example usage: load a trained agent (after training in train.py) and evaluate
if __name__ == "__main__":
    # Assuming agent saved or accessible
    agent = PrimalDualPG()
    # ... (load trained parameters)
    p = 4
    avg_ret, risk = evaluate_policy(agent, p)
    print(f"Avg return: {avg_ret}, Mean-{p} risk: {risk}")
