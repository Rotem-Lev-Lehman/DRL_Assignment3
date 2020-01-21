import copy

import gym
import numpy as np
import collections
from Proggressive_Stuff import *

env_name = "MountainCarContinuous-v0"
part_name = "Part1"
env = gym.make(env_name)
exp_ind = 10
np.random.seed(1)
actor_critic = True
baseline = True
use_trained_network = False
model_saving_name = env_name + "_" + str(use_trained_network) + '_' + str(exp_ind)
render = False
start_time = time.time()
rewards_num = 0
cart_pole_satisfying_avg = 475
acrobot_satisfying_avg = -85
EPISODES_TO_HELP_NETWORK = 30
mountainCar_satisfying_avg = 75
satisfying_average = mountainCar_satisfying_avg

max_state_size = 4
max_action_size = 2

model1_loading_name = "CartPole-v1_1"
state_size1 = 4
action_size1 = 2

model2_loading_name = "Acrobot-v1_False_1"
state_size2 = 6
action_size2 = 3

# Define hyper parameters
state_size = env.observation_space.shape[0]
state_size_for_reshape = 4
if type(env.action_space) == gym.spaces.box.Box:
    action_size = 2
else:
    action_size = env.action_space.n

max_episodes = 5000
max_steps = env._max_episode_steps
max_steps_for_first_episode = 10000
decay_max_steps_by = 250
discount_factor = 0.99
learning_rate = 0.0004

# Initialize the policy network
tf.reset_default_graph()

policy_network1 = PolicyNetworkAlreadyTrained(state_size1, action_size1, learning_rate, model1_loading_name)
value_network1 = ValueNetworkAlreadyTrained(state_size1, learning_rate, model1_loading_name)

policy_network2 = PolicyNetworkAlreadyTrained(state_size2, action_size2, learning_rate, model2_loading_name)
value_network2 = ValueNetworkAlreadyTrained(state_size2, learning_rate, model2_loading_name)

policy = PolicyNetwork(max_state_size, max_action_size, learning_rate, policy_network1, policy_network2,
                       model_saving_name)
value = ValueNetwork(max_state_size, learning_rate, value_network1, value_network2)


def reshape_state(state, policy, flag = True):
    a = copy.deepcopy(state)
    reshape_size = state_size_for_reshape if flag else state_size
    for i in range(policy.state_size - reshape_size):
        state = np.append(state, 0)
    return state.reshape((1, policy.state_size))


def reshape_action(actions_distribution):
    for i in range(policy.action_size - action_size):
        actions_distribution = np.delete(actions_distribution, -1)
    sump = np.sum(actions_distribution)
    for i in range(len(actions_distribution)):
        actions_distribution[i] /= sump
    return actions_distribution


# save the current model to use it later:
saver = tf.compat.v1.train.Saver()

# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    read_to_networks(policy_network1, value_network1, 'weights/' + model1_loading_name + '.csv', sess)
    read_to_networks(policy_network2, value_network2, 'weights/' + model2_loading_name + '.csv', sess)

    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0

    reward_for_breaking_record = 1.5

    first_episode = True

    for episode in range(max_episodes):
        policy.tensorboard.step = episode
        state = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []
        most_left = state[0][0]
        most_right = state[0][0]

        current_max_steps = max(max_steps, max_steps_for_first_episode)
        max_steps_for_first_episode -= decay_max_steps_by
        env._max_episode_steps = current_max_steps

        for step in range(current_max_steps):
            state = reshape_state(state, policy, False)
            if first_episode:
                actions_distribution = np.ones(policy.action_size)
            else:
                actions_distribution = sess.run(policy.actions_distribution,
                                                {policy.state: state,
                                                 policy_network1.state: reshape_state(state, policy_network1),
                                                 policy_network2.state: reshape_state(state, policy_network2)})
            actions_distribution = reshape_action(actions_distribution)
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)

            if env_name is "MountainCarContinuous-v0":
                if action == 0:
                    action_for_env = [-1]
                else:
                    action_for_env = [1]
            next_state, reward, done, _ = env.step(action_for_env)
            next_state = next_state.reshape([1, state_size])
            next_state_to_transition = reshape_state(next_state, policy, False)

            if render:
                env.render()

            action_one_hot = np.zeros(max_action_size)
            action_one_hot[action] = 1
            if not done and episode < EPISODES_TO_HELP_NETWORK:
                if next_state[0][0] < most_left:
                    most_left = next_state[0][0]
                    reward_for_transition = reward_for_breaking_record
                elif next_state[0][0] > most_right:
                    most_right = next_state[0][0]
                    reward_for_transition = reward_for_breaking_record
                else:
                    reward_for_transition = reward
            else:
                reward_for_transition = reward
            '''
            if done and episode < EPISODES_TO_HELP_NETWORK and step < current_max_steps - 1:
                reward_for_transition = current_max_steps
            else:
                reward_for_transition = reward
            '''
            episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward_for_transition,
                                                  next_state=next_state_to_transition, done=done))
            episode_rewards[episode] += reward

            if done:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                else:
                    average_rewards = np.mean(episode_rewards[:episode + 1])
                policy.tensorboard.update_stats(last_100_average_reward=average_rewards,
                                                reward=episode_rewards[episode])
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode,
                                                                                   round(episode_rewards[episode], 2),
                                                                                   round(average_rewards, 2)))
                rewards_num += episode_rewards[episode]
                if episode > 98 and average_rewards > satisfying_average:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break

            state = next_state

        if solved:
            saver.save(sess=sess, save_path="models/" + part_name + "/" + model_saving_name)
            break

        # Compute Rt for each time-step t and update the network's weights
        # update the weights of the two networks if we are using actor critic
        for t, transition in enumerate(episode_transitions):
            if actor_critic:
                total_discounted_return = sum(
                    discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt
                if transition.done:
                    next_state_value = 0
                else:
                    next_state_value = sess.run(value.value, {value.state: transition.next_state,
                                                              value_network1.state: reshape_state(transition.next_state,
                                                                                                  value_network1),
                                                              value_network2.state: reshape_state(transition.next_state,
                                                                                                  value_network2)})
                target_for_value = total_discounted_return
                delta = target_for_value - sess.run(value.value, {value.state: transition.state,
                                                                  value_network1.state: reshape_state(transition.state,
                                                                                                      value_network1),
                                                                  value_network2.state: reshape_state(transition.state,
                                                                                                      value_network2)})
                value_dict = {value.state: transition.state, value.R_t: delta,
                              value_network1.state: reshape_state(transition.state, value_network1),
                              value_network2.state: reshape_state(transition.state, value_network2)}
                _, value_loss = sess.run([value.optimizer, value.loss], value_dict)
                feed_dict = {policy.state: transition.state, policy.R_t: delta, policy.action: transition.action,
                             policy_network1.state: reshape_state(transition.state, policy_network1),
                             policy_network2.state: reshape_state(transition.state, policy_network2)}
                _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
                policy.tensorboard.update_stats(loss=loss)
            else:
                total_discounted_return = sum(
                    discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt
                if baseline:
                    total_discounted_return -= sess.run(value.value, {value.state: transition.state})
                    value_dict = {value.state: transition.state, value.R_t: total_discounted_return}
                    _, value_loss = sess.run([value.optimizer, value.loss], value_dict)
                feed_dict = {policy.state: transition.state, policy.R_t: total_discounted_return,
                             policy.action: transition.action}
                _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
                policy.tensorboard.update_stats(loss=loss)

        first_episode = False

print()
total_time = time.time() - start_time
policy.tensorboard.update_stats(Overall_time=total_time, Time_per_reward=total_time / rewards_num,
                                Time_per_episode=total_time / episode)
