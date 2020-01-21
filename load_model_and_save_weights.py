import time
import gym
import numpy as np
import tensorflow as tf
import collections
import csv
from ModifiedTensorBoard import ModifiedTensorBoard

env_name = "MountainCarContinuous-v0"
part_name = "Part1"
env = gym.make(env_name)
exp_ind = 2
model_loading_name = "MountainCarContinuous-v0_False_2"
file_name = "weights/" + model_loading_name + ".csv"
np.random.seed(1)
actor_critic = True
baseline = True
use_trained_network = True
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
            self.W2 = tf.get_variable("W2", [12, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [12], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3", [12, self.action_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.add(tf.matmul(self.Z1, self.W2), self.b2)
            self.output = tf.add(tf.matmul(self.Z1, self.W3), self.b3)

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

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [12], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3", [12, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b3 = tf.get_variable("b3", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.add(tf.matmul(self.Z1, self.W2), self.b2)
            self.output = tf.add(tf.matmul(self.Z1, self.W3), self.b3)

            # Softmax probability distribution over actions
            self.value = self.output
            # Loss with negative log probability
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
max_steps_for_first_episode = 10000
decay_max_steps_by = 250
discount_factor = 0.99
learning_rate = 0.0004

# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(max_state_size, max_action_size, learning_rate)
value = ValueNetwork(max_state_size, learning_rate)


# save the current model to use it later:
saver = tf.compat.v1.train.Saver()

# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    # restore previous session:
    saver.restore(sess=sess, save_path="models/" + part_name + "/" + model_loading_name)

    variables = tf.trainable_variables()


    with open(file_name, newline='', mode='w') as output_file:
        weights_writer = csv.writer(output_file, delimiter=';')

        for curr in variables:
            #get the values of the previous session:
            val = sess.run(curr)
            stringVal = str(val).replace("\n","")
            while "  " in stringVal:
                stringVal = stringVal.replace("  "," ")
            stringVal = stringVal.replace(" ]","]").replace("[ ","[").replace(" ",",")
            name = curr.name
            weights_writer.writerow([name, stringVal])

print("done")
