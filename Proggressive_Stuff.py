import csv
import time

import tensorflow as tf

from ModifiedTensorBoard import ModifiedTensorBoard


class PolicyNetworkAlreadyTrained:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = 'policy_' + name

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
    def __init__(self, state_size, action_size, learning_rate, policy_network1, policy_network2, model_saving_name, name='policy_network'):
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
            # Loss with negative log probability
            self.loss = tf.reduce_mean(tf.square(self.R_t - self.value))  # loss = mse
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


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