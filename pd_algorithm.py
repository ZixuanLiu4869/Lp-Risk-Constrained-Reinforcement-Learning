import numpy as np
import torch

from environment import GRID_H, GRID_W, START_STATE, step_env

class PrimalDualPG:
    def __init__(self, lr_pi=0.01, lr_lambda=0.005, gamma=1.0):
        self.gamma = gamma
        # Policy parameter: logits for each state-action pair
        self.logits = torch.nn.Parameter(torch.zeros(GRID_H * GRID_W, 4))
        # Lagrange multiplier (dual variable for risk constraint)
        self.lambda_param = 0.0
        self.lr_pi = lr_pi
        self.lr_lambda = lr_lambda

    def select_action(self, state):
        """Select action according to current policy (stochastic). Returns action and log-probability."""
        state_idx = state[0] * GRID_W + state[1]
        probs = torch.softmax(self.logits[state_idx], dim=-1)
        action = int(torch.distributions.Categorical(probs).sample().item())
        log_prob = torch.log(probs[action] + 1e-8)
        return action, log_prob

    def run_episode(self):
        """Simulate one episode using current policy. Returns list of (log_prob, cost) for each episode."""
        state = START_STATE
        log_probs = []
        total_cost = 0.0
        step_count = 0
        # Run until termination or step limit
        while True:
            action, log_prob = self.select_action(state)
            next_state, reward, done = step_env(state, action)
            log_probs.append(log_prob)
            # Accumulate cost (negative reward)
            total_cost += -reward
            state = next_state
            step_count += 1
            if done or step_count > 100:  # safety break after 100 steps
                break
        return log_probs, total_cost

    def train(self, risk_threshold, p, num_iterations=1000, batch_size=50):
        """Train policy with primal-dual updates for given risk threshold and Lp value."""
        cost_history = []
        risk_history = []
        lambda_history = []
        for it in range(num_iterations):
            batch_log_probs = []
            batch_costs = []
            # Collect trajectories
            for _ in range(batch_size):
                log_probs, total_cost = self.run_episode()
                batch_log_probs.append(log_probs)
                batch_costs.append(total_cost)
            # Compute average cost and risk (mean-Lp)
            avg_cost = np.mean(batch_costs)
            avg_cost_p = np.mean([c**p for c in batch_costs])
            risk = (avg_cost_p ** (1.0/p)) if avg_cost_p > 0 else 0.0
            cost_history.append(avg_cost); risk_history.append(risk); lambda_history.append(self.lambda_param)
            # Primal update: policy gradient ascent
            # Compute policy gradient objective: J = -E[C] - λ * (E[C^p] - C_thresh^p)  (via REINFORCE)
            # We approximate gradients using sampled trajectories.
            # Construct loss = (C + λ * C^p) as negative of J for gradient *descent*
            loss = 0.0
            baseline = -avg_cost - self.lambda_param * (avg_cost ** p)  # baseline uses batch averages (optional)
            for log_probs, total_cost in zip(batch_log_probs, batch_costs):
                # pseudo-reward for policy = -C - λ * C^p
                R_tilde = -total_cost - self.lambda_param * (total_cost ** p)
                advantage = R_tilde - baseline
                # Sum log-probs over episode
                episode_log_prob = torch.stack(log_probs).sum()
                loss += -advantage * episode_log_prob  # negative for gradient descent
            loss = loss / batch_size
            # Backpropagate and update policy parameters
            loss.backward()
            with torch.no_grad():
                self.logits -= self.lr_pi * self.logits.grad
            self.logits.grad = None
            # Dual update: adjust λ (gradient ascent on λ to enforce constraint)
            constraint_violation = risk - risk_threshold
            self.lambda_param += self.lr_lambda * constraint_violation
            if self.lambda_param < 0.0:
                self.lambda_param = 0.0  # enforce λ >= 0
        return cost_history, risk_history, lambda_history
