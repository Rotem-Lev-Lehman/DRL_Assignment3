import time
import gym
import numpy as np
import tensorflow as tf
import collections
from ModifiedTensorBoard import ModifiedTensorBoard

env_name = "Acrobot-v1"
part_name = "Part1"
env = gym.make(env_name)
exp_ind = 2
model_loading_name = "Acrobot-v1_False_1"
np.random.seed(1)
actor_critic = True
baseline = True
use_trained_network = True
model_saving_name = env_name + "_" + str(use_trained_network) + '_' + str(exp_ind)
render = False
start_time = time.time()
rewards_num = 0
cart_pole_satisfying_avg = 475
acrobot_satisfying_avg = -100
EPISODES_TO_HELP_NETWORK = 30
satisfying_average = acrobot_satisfying_avg

max_state_size = 6
max_action_size = 3


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(model_saving_name, int(time.time())))

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 12],initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [12], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3", [12, self.action_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer())


            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.add(tf.matmul(self.Z1, self.W2), self.b2)
            self.output = tf.add(tf.matmul(self.A1, self.W3), self.b3)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueNetwork:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.name = name
        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [12], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3", [12, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.add(tf.matmul(self.Z1, self.W2), self.b2)
            self.output = tf.add(tf.matmul(self.A1, self.W3), self.b3)

            # Softmax probability distribution over actions
            self.value = self.output
            # Loss with negative log probability
            self.loss = tf.reduce_mean(tf.square(self.R_t - self.value))  # loss = mse
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def set_weights(self, other):
        self.W1 = other.W1
        self.b1 = other.b1
        self.W2 = other.W2
        self.b2 = other.b2


# Define hyper parameters
state_size = env.observation_space.shape[0]
if type(env.action_space) == gym.spaces.box.Box:
    action_size = 2
else:
    action_size = env.action_space.n

max_episodes = 5000
max_steps = env._max_episode_steps
max_steps_for_first_episode = 10000
discount_factor = 0.99
learning_rate = 0.0004

# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(max_state_size, max_action_size, learning_rate)
value = ValueNetwork(max_state_size, learning_rate)


def reshape_state(state):
    for i in range(policy.state_size - state_size):
        state = np.append(state, 0)
    return state.reshape((1,policy.state_size))


def reshape_action(actions_distribution):
    for i in range(policy.action_size - action_size):
        actions_distribution = np.delete(actions_distribution,-1)
    sump = np.sum(actions_distribution)
    for i in range(len(actions_distribution)):
        actions_distribution[i] /= sump
    return actions_distribution

# save the current model to use it later:
saver = tf.compat.v1.train.Saver()

# Start training the agent with REINFORCE algorithm
def calculate_total_discount_return_for_step(episode_transitions):
    returns = []
    total_return_until_now = 0
    for i in range(len(episode_transitions) - 1, -1, -1):
        total_return_until_now = total_return_until_now * discount_factor + episode_transitions[i].reward
        returns.insert(0, total_return_until_now)
        #sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt
    return returns


with tf.Session() as sess:
    if use_trained_network:
        # restore previous session:
        saver.restore(sess=sess, save_path="models/" + part_name + "/" + model_loading_name)

    else:
        # initialize new networks:
        sess.run(tf.global_variables_initializer())

    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0

    first_episode = not use_trained_network


    for episode in range(max_episodes):
        policy.tensorboard.step = episode
        state = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []
        s = env.state
        max_height_till_now = (-np.cos(s[0]) - np.cos(s[1] + s[0]))
        total_rewards_for_episode = 0

        current_max_steps = max_steps_for_first_episode if first_episode else max_steps
        env._max_episode_steps = current_max_steps

        for step in range(current_max_steps):
            state = reshape_state(state)
            if first_episode:
                actions_distribution = np.ones(policy.action_size)
            else:
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})

            actions_distribution = reshape_action(actions_distribution)
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])
            next_state_to_transition = reshape_state(next_state)

            if render:
                env.render()

            action_one_hot = np.zeros(max_action_size)
            action_one_hot[action] = 1
            s = env.state
            height = (-np.cos(s[0]) - np.cos(s[1] + s[0]))
            reward_for_transition = reward
            if done:
                reward_for_transition += height*(current_max_steps-step)
            """if done and first_episode and step < current_max_steps - 1:
                reward_for_transition = 1000
                print (height)
            elif not first_episode:

                #reward_for_transition = reward

                reward_for_transition = height
                #if height > max_height_till_now:
                #    max_height_till_now = height
                #    reward_for_transition += 0.5

                #if done and step < current_max_steps - 1:
                #    reward_for_transition += 100
                
                #reward_for_transition += (-np.cos(s[0]) - np.cos(s[1] + s[0]))
                #reward_for_transition += -np.cos(s[0])
            else:
                reward_for_transition = reward
            """
            total_rewards_for_episode += reward_for_transition
            episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward_for_transition, next_state=next_state_to_transition, done=done))
            episode_rewards[episode] += reward

            if done:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                else:
                    average_rewards = np.mean(episode_rewards[:episode+1])
                policy.tensorboard.update_stats(last_100_average_reward=average_rewards, reward=episode_rewards[episode])
                print("Episode {} Reward: {} Trans_reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], total_rewards_for_episode, round(average_rewards, 2)))
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
        total_discounted_return_for_step = calculate_total_discount_return_for_step(episode_transitions)
        for t, transition in enumerate(episode_transitions):
            if actor_critic:
                #total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt
                total_discounted_return = total_discounted_return_for_step[t]
                if transition.done:
                    next_state_value = 0
                else:
                    next_state_value = sess.run(value.value, {value.state: transition.next_state})
                target_for_value = total_discounted_return
                delta = target_for_value - sess.run(value.value, {value.state: transition.state})
                value_dict = {value.state: transition.state, value.R_t: delta}
                _, value_loss = sess.run([value.optimizer, value.loss], value_dict)
                feed_dict = {policy.state: transition.state, policy.R_t: delta, policy.action: transition.action}
                _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
                policy.tensorboard.update_stats(loss=loss)
            else:
                total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:])) # Rt
                if baseline:
                    total_discounted_return -= sess.run(value.value, {value.state: transition.state})
                    value_dict = {value.state: transition.state, value.R_t: total_discounted_return}
                    _, value_loss = sess.run([value.optimizer, value.loss], value_dict)
                feed_dict = {policy.state: transition.state, policy.R_t: total_discounted_return, policy.action: transition.action}
                _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
                policy.tensorboard.update_stats(loss=loss)

        first_episode = False

print()
total_time = time.time()-start_time
policy.tensorboard.update_stats(Overall_time=total_time, Time_per_reward=total_time/rewards_num, Time_per_episode=total_time/episode)
