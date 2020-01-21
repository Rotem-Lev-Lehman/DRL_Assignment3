import csv
import time
import gym
import numpy as np
import tensorflow as tf
import collections
from ModifiedTensorBoard import ModifiedTensorBoard

env_name = "CartPole-v1"
part_name = "Part1"
env = gym.make(env_name)
exp_ind = 11
model1_loading_name = "Acrobot-v1_False_1"
model2_loading_name = "MountainCarContinuous-v0_False_2"
state_size1 = 6
action_size1 = 3

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


class PolicyNetworkAlreadyTrained:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = 'policy_' + name
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(model_saving_name, int(time.time())))

        with tf.variable_scope(self.name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
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


class ValueNetworkAlreadyTrained:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.name = 'value_' + name
        with tf.variable_scope(self.name ):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
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
            #self.value = tf.squeeze(tf.nn.relu(self.output))
            # Loss with negative log probability
            #self.loss = tf.reduce_mean(self.output * self.R_t)
            self.loss = tf.reduce_mean(tf.square(self.R_t - self.value))  # loss = mse
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, policy_network1, policy_network2, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(model_saving_name, int(time.time())))

        with tf.variable_scope(self.name ):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [36, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [12], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3", [36, self.action_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.concatZ1 = tf.concat([self.Z1, policy_network1.Z1, policy_network2.Z1], 1)
            self.A1 = tf.add(tf.matmul(self.concatZ1, self.W2), self.b2)
            self.concatA1 = tf.concat([self.A1, policy_network1.A1, policy_network2.A1], 1)
            self.output = tf.add(tf.matmul([self.concatA1], self.W3), self.b3)


            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueNetwork:
    def __init__(self, state_size, learning_rate, value_network1, value_network2, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.name = name
        with tf.variable_scope(self.name ):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [36, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [12], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3", [36, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.concatZ1 = tf.concat([self.Z1, value_network1.Z1, value_network2.Z1], 1)
            self.A1 = tf.add(tf.matmul(self.concatZ1, self.W2), self.b2)
            self.concatA1 = tf.concat([self.A1, value_network1.A1, value_network2.A1], 1)
            self.output = tf.add(tf.matmul(self.concatA1, self.W3), self.b3)

            # Softmax probability distribution over actions
            self.value = self.output
            #self.value = tf.squeeze(tf.nn.relu(self.output))
            # Loss with negative log probability
            #self.loss = tf.reduce_mean(self.output * self.R_t)
            self.loss = tf.reduce_mean(tf.square(self.R_t - self.value))  # loss = mse
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


# Define hyper parameters
state_size = env.observation_space.shape[0]
if type(env.action_space) == gym.spaces.box.Box:
    action_size = 2
else:
    action_size = env.action_space.n

max_episodes = 5000
max_steps = env._max_episode_steps
#max_steps_for_first_episode = max_steps
discount_factor = 0.99
learning_rate = 0.0004

# Initialize the policy network
tf.reset_default_graph()


def read_to_networks(policy_network, value_network, path, sess):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            name = row[0]
            value = row[1]
            if 'policy' in name:
                if 'W1' in name:
                    theChoosenOne = policy_network.W1
                elif 'W2' in name:
                    theChoosenOne = policy_network.W2
                elif 'W3' in name:
                    theChoosenOne = policy_network.W3
                elif 'b1' in name:
                    theChoosenOne = policy_network.b1
                elif 'b2' in name:
                    theChoosenOne = policy_network.b2
                elif 'b3' in name:
                    theChoosenOne = policy_network.b3
            elif 'value' in name:
                if 'W1' in name:
                    theChoosenOne = value_network.W1
                elif 'W2' in name:
                    theChoosenOne = value_network.W2
                elif 'W3' in name:
                    theChoosenOne = value_network.W3
                elif 'b1' in name:
                    theChoosenOne = value_network.b1
                elif 'b2' in name:
                    theChoosenOne = value_network.b2
                elif 'b3' in name:
                    theChoosenOne = value_network.b3
            assign_opt = theChoosenOne.assign(parse_value(value))
            sess.run(assign_opt)
            theChoosenOne = tf.stop_gradient(theChoosenOne)


def parse_value(string):
    values = string.split("[")
    if len(values) == 2:
        return parse_line(values[1][:-1])
    else:
        ans = []
        for value in values[2:]:
            ans.append(parse_line(value[:-2]))
        return ans


def parse_line(string):
    ans = []
    for num in string.split(','):
        if len(num) > 0:
            ans.append(float(num))
    return ans


policy_network1 = PolicyNetworkAlreadyTrained(state_size1, action_size1, learning_rate, model1_loading_name)
value_network1 = ValueNetworkAlreadyTrained(state_size1, learning_rate, model1_loading_name)

policy_network2 = PolicyNetworkAlreadyTrained(state_size2, action_size2, learning_rate, model2_loading_name)
value_network2 = ValueNetworkAlreadyTrained(state_size2, learning_rate, model2_loading_name)

policy = PolicyNetwork(max_state_size, max_action_size, learning_rate, policy_network1, policy_network2)
value = ValueNetwork(max_state_size, learning_rate, value_network1, value_network2)


def reshape_state(state,policy):
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
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    read_to_networks(policy_network1, value_network1, 'weights/' + model1_loading_name + '.csv',sess)
    read_to_networks(policy_network2, value_network2, 'weights/' + model2_loading_name + '.csv',sess)

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
            state = reshape_state(state,policy)
            actions_distribution = sess.run(policy.actions_distribution, {policy.state: reshape_state(state,policy), policy_network1.state: reshape_state(state,policy_network1), policy_network2.state: reshape_state(state,policy_network2)})
            #actions_distribution = reshape_action(actions_distribution)
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
                    average_rewards = np.mean(episode_rewards[:episode+1])
                policy.tensorboard.update_stats(last_100_average_reward=average_rewards, reward=episode_rewards[episode])
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
                total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt
                if transition.done:
                    next_state_value = 0
                else:
                    next_state_value = sess.run(value.value, {value.state: reshape_state(transition.next_state,value), value_network1.state: reshape_state(transition.next_state,value_network1), value_network2.state: reshape_state(transition.next_state,value_network2)})
                target_for_value = total_discounted_return
                delta = target_for_value - sess.run(value.value, {value.state: reshape_state(transition.state,value), value_network1.state: reshape_state(transition.state,value_network1), value_network2.state: reshape_state(transition.state,value_network2)})
                value_dict = {value.state: transition.state, value.R_t: delta, value_network1.state: reshape_state(transition.state,value_network1), value_network2.state: reshape_state(transition.state,value_network2)}
                _, value_loss = sess.run([value.optimizer, value.loss], value_dict)
                feed_dict = {policy.state: transition.state, policy.R_t: delta, policy.action: transition.action,policy_network1.state: reshape_state(transition.state,policy_network1), policy_network2.state: reshape_state(transition.state,policy_network2)}
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
