import numpy as np


def reshape_state(state, policy, state_size):
    for i in range(policy.state_size - state_size):
        state = np.append(state, 0)
    return state.reshape((1, policy.state_size))


def reshape_action(actions_distribution, policy, action_size):
    for i in range(policy.action_size - action_size):
        actions_distribution = np.delete(actions_distribution, -1)
    sump = np.sum(actions_distribution)
    for i in range(len(actions_distribution)):
        actions_distribution[i] /= sump
    return actions_distribution