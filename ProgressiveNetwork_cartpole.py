import gym
import numpy as np
import collections
from Proggressive_Stuff import *

env_name = "CartPole-v1"
part_name = "Part1"
env = gym.make(env_name)
exp_ind = 11

model1_loading_name = "Acrobot-v1_False_1"
state_size1 = 6
action_size1 = 3

model2_loading_name = "MountainCarContinuous-v0_False_2"
state_size2 = 4
action_size2 = 2

np.random.seed(1)
actor_critic = True
baseline = True
use_trained_network = False
render = False
start_time = time.time()
rewards_num = 0
cart_pole_satisfying_avg = 475
acrobot_satisfying_avg = -85
satisfying_average = cart_pole_satisfying_avg
training_acrobotNet = False
training_mountainNet = False

if training_acrobotNet:
    max_state_size = 6
    max_action_size = 3
    model_saving_name = env_name + "_" + 'acrobot' + str(exp_ind)
elif training_mountainNet:
    max_state_size = 4
    max_action_size = 2
    model_saving_name = env_name + "_" + 'mountain' + str(exp_ind)
else:
    max_state_size = 4
    max_action_size = 2
    model_saving_name = env_name + "_" + str(exp_ind)

# Define hyper parameters
state_size = env.observation_space.shape[0]
if type(env.action_space) == gym.spaces.box.Box:
    action_size = 2
else:
    action_size = env.action_space.n

max_episodes = 5000
max_steps = env._max_episode_steps
# max_steps_for_first_episode = max_steps
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


def reshape_state(state, policy):
    for i in range(policy.state_size - state_size):
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
    for episode in range(max_episodes):
        policy.tensorboard.step = episode
        state = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []

        for step in range(max_steps):
            state = reshape_state(state, policy)
            actions_distribution = sess.run(policy.actions_distribution, {policy.state: reshape_state(state, policy),
                                                                          policy_network1.state: reshape_state(state,
                                                                                                               policy_network1),
                                                                          policy_network2.state: reshape_state(state,
                                                                                                               policy_network2)})
            action_chosen = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            if training_acrobotNet:
                while step == 0 and action_chosen == 1:
                    action_chosen = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                if action_chosen == 1:
                    reward_for_transition, done = -1, False
                    reward = 0
                else:
                    if action_chosen == 2:
                        action = 1
                    else:
                        action = action_chosen
                    next_state, reward, done, _ = env.step(action)
                    next_state = next_state.reshape([1, state_size])
                    next_state_to_transition = reshape_state(next_state, policy)
                    reward_for_transition = reward
            else:
                action = action_chosen
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])
                next_state_to_transition = reshape_state(next_state, policy)
                reward_for_transition = reward
            if render:
                env.render()

            action_one_hot = np.zeros(max_action_size)
            action_one_hot[action_chosen] = 1

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
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
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
                    next_state_value = sess.run(value.value, {value.state: reshape_state(transition.next_state, value),
                                                              value_network1.state: reshape_state(transition.next_state,
                                                                                                  value_network1),
                                                              value_network2.state: reshape_state(transition.next_state,
                                                                                                  value_network2)})
                target_for_value = total_discounted_return
                delta = target_for_value - sess.run(value.value, {value.state: reshape_state(transition.state, value),
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
