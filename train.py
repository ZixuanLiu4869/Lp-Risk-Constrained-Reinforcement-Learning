import numpy as np
from pd_algorithm import PrimalDualPG
from dp_algorithm import solve_constrained_MDP

# Define Lp values to test
p_values = [1, 2, 4, 8]
risk_threshold = 5.0  # allowed mean-Lp cost (set to cost of safe path)

# Storage for results
results = { "p": [], "return_pd": [], "risk_pd": [], "return_dp": [], "risk_dp": [] }

for p in p_values:
    print(f"=== Evaluating for p = {p} ===")
    # Train primal-dual policy gradient
    agent = PrimalDualPG(lr_pi=0.05, lr_lambda=0.05)
    cost_hist, risk_hist, lambda_hist = agent.train(risk_threshold=risk_threshold, p=p,
                                                   num_iterations=500, batch_size=50)
    # Estimate performance of trained policy
    # (We simulate episodes to estimate average return and risk)
    returns = []
    costs = []
    for _ in range(1000):
        _, total_cost = agent.run_episode()
        costs.append(total_cost)
        returns.append(-total_cost)
    avg_return_pd = np.mean(returns)
    risk_pd = (np.mean(np.array(costs) ** p)) ** (1.0/p)
    print(f"Primal-dual: avg return = {avg_return_pd:.3f}, risk = {risk_pd:.3f}, lambda = {lambda_hist[-1]:.3f}")

    # Solve via DP (model-based approach)
    policy_dp, avg_return_dp, risk_dp = solve_constrained_MDP(risk_threshold, p)
    print(f"DP solution: avg return = {avg_return_dp:.3f}, risk = {risk_dp:.3f}")

    results["p"].append(p)
    results["return_pd"].append(avg_return_pd)
    results["risk_pd"].append(risk_pd)
    results["return_dp"].append(avg_return_dp)
    results["risk_dp"].append(risk_dp)
