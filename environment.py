import numpy as np
import random

# GridWorld dimensions and special states
GRID_H, GRID_W = 5, 5
START_STATE = (0, 0)
GOAL_STATE = (4, 0)
HAZARD_STATES = {(2, 0)}  # set of hazardous states

# Rewards/penalties
STEP_REWARD = -1.0        # normal step cost
HAZARD_PENALTY = -10.0    # penalty for entering hazard
GOAL_REWARD = 0.0         # reward for reaching goal

# Probability of hazard event (falling into hazardous state)
HAZARD_FAIL_PROB = 0.1

def step_env(state, action):
    """
    Apply action in the grid. Actions: 0=up,1=right,2=down,3=left.
    Returns: next_state, reward, done.
    """
    x, y = state
    if action == 0:    nx, ny = x - 1, y       # up
    elif action == 1:  nx, ny = x, y + 1       # right
    elif action == 2:  nx, ny = x + 1, y       # down
    elif action == 3:  nx, ny = x, y - 1       # left
    else:              raise ValueError("Invalid action")

    # Check bounds
    if nx < 0 or nx >= GRID_H or ny < 0 or ny >= GRID_W:
        # Invalid move (off-grid): stay and incur step cost
        next_state = state
        reward = STEP_REWARD
        done = False
    else:
        next_state = (nx, ny)
        # Check for terminal or hazardous transitions
        if next_state == GOAL_STATE:
            reward = GOAL_REWARD    # reaching goal yields 0 (no step cost)
            done = True
        elif next_state in HAZARD_STATES:
            # Hazard entry: apply penalty with probability, otherwise just step cost
            if random.random() < HAZARD_FAIL_PROB:
                reward = HAZARD_PENALTY
            else:
                reward = STEP_REWARD
            done = False  # not terminating; agent continues even after penalty
        else:
            # Normal transition
            reward = STEP_REWARD
            done = False
    return next_state, reward, done
