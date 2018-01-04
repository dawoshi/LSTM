""" LSTM with using a BasicLSTMCell """

import tensorflow as tf

class Model:
    def __init__(self):
        tf.set_random_seed(0)
        self.loss = 0
        self.inputs = None
        self.labels = None
        self.optimizer = None
        self.session = tf.Session()
        self.lstm_cell = None
        self.build_graph()

    def build_graph(self):
        time_steps = 2
        batch_size = 1
        num_inputs = 2
        num_units = 1
        learning_rate = 0.1

        self.inputs = tf.placeholder(tf.float32, [time_steps, batch_size, num_inputs])
        self.labels = tf.placeholder(tf.float32, [time_steps, batch_size, 1])

        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units, 0)
        labels_unstacked = tf.unstack(self.labels, axis=0)
        inputs_unstacked = tf.unstack(self.inputs, axis=0)

        outputs, states = tf.nn.static_rnn(self.lstm_cell, inputs_unstacked, dtype=tf.float32)

        for output, label in zip(outputs, labels_unstacked):
            self.loss += tf.square(output - label)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def train(self, inputs, labels):
        loss, _ = self.session.run([self.loss, self.optimizer],
                                                 feed_dict={
                                                     self.inputs: inputs,
                                                     self.labels: labels
                                                 })
        return loss