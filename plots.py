import matplotlib.pyplot as plt
import numpy as np

# Example data (to be replaced with actual results from train.py)
p_vals = np.array([1, 2, 4, 8])
avg_returns = np.array([-3.9, -3.9, -5.0, -5.0])   # average returns for each p
risks = np.array([3.9, 4.74, 5.0, 5.0])            # corresponding mean-Lp risks

# Plot risk-return tradeoff
plt.figure()
plt.plot(p_vals, avg_returns, marker='o', label='Average Return')
plt.plot(p_vals, -risks, marker='s', label='-Mean-$L_p$ Cost')
plt.xlabel('$p$ (risk order)')
plt.ylabel('Value')
plt.title('Final Return vs Risk for different $p$')
plt.legend()
plt.savefig('risk_return_comparison.png')
plt.close()

# Plot policy heatmaps for each p
# (Here we manually construct example optimal policies for demonstration)
# 0:up, 1:right, 2:down, 3:left
policies = {
    1: {(0,0):2, (1,0):2, (2,0):2, (3,0):2},        # hazard path (down every step)
    2: {(0,0):2, (1,0):2, (2,0):2, (3,0):2},        # hazard path
    4: {(0,0):2, (1,0):1, (1,1):2, (2,1):2, (3,1):3, (3,0):2},  # safe detour path
    8: {(0,0):2, (1,0):1, (1,1):2, (2,1):2, (3,1):3, (3,0):2}   # safe detour path
}
fig, axs = plt.subplots(2, 2, figsize=(6,6))
arrow = {0:'↑', 1:'→', 2:'↓', 3:'←'}
for idx, p in enumerate([1, 2, 4, 8]):
    ax = axs[idx//2, idx%2]
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(True)
    # Shade hazard and goal
    hx, hy = (2,0)
    gx, gy = (4,0)
    ax.add_patch(plt.Rectangle((hy-0.5, hx-0.5), 1, 1, color='red', alpha=0.3))
    ax.add_patch(plt.Rectangle((gy-0.5, gx-0.5), 1, 1, color='green', alpha=0.3))
    # Mark H, G, S
    ax.text(0, 2, 'H', color='red', ha='center', va='center', fontweight='bold')
    ax.text(0, 4, 'G', color='green', ha='center', va='center', fontweight='bold')
    ax.text(0, 0, 'S', color='blue', ha='left', va='top')
    # Draw policy arrows
    for (x,y), a in policies[p].items():
        if (x,y) == (2,0) or (x,y) == (4,0):  # skip hazard and goal markers
            continue
        ax.text(y, x, arrow[a], ha='center', va='center', color='black')
    ax.set_title(f'$p={p}$')
plt.tight_layout()
plt.savefig('policy_heatmaps.png')
plt.close()
