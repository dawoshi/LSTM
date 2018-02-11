""" LSTM without using as BasicLSTMCell """

import tensorflow as tf

state_weights = tf.get_variable("state_weights", dtype=tf.float32, initializer=tf.constant(
    [
        [0.26770878, 0.59700513, 0.33730936, -0.92146528],
        [0.16705453, 0.16668344, -0.220312, -0.4016704]
    ]
))

hidden_state_weights = tf.get_variable("hidden_state_weights", dtype=tf.float32, initializer=tf.constant(
    [[0.73747325, -0.81513613, 0.33634782, 0.90510082]]
))

bias = tf.get_variable("bias", dtype=tf.float32, initializer=tf.constant([0.0, 0.0, 0.0, 0.0]))

class Model:
    def __init__(self):
        tf.set_random_seed(0)
        self.loss = 0
        self.inputs = None
        self.labels = None
        self.optimizer = None
        self.session = tf.Session()
        self.build_graph()

    def lstm(self, input, hidden_state, state):
        gate_inputs = tf.matmul(input, state_weights) + tf.matmul(hidden_state, hidden_state_weights) + bias
        input_gate, new_input, forget_gate, output_gate = tf.split(axis=1, num_or_size_splits=4, value=gate_inputs)
        new_state = tf.tanh(new_input) * tf.sigmoid(input_gate) + tf.sigmoid(forget_gate) * state
        new_hidden_state = tf.tanh(new_state) * tf.sigmoid(output_gate)
        return new_hidden_state, new_state

    def build_graph(self):
        time_steps = 2
        batch_size = 1
        num_inputs = 2
        learning_rate = 0.1

        self.inputs = tf.placeholder(tf.float32, [time_steps, batch_size, num_inputs])
        self.labels = tf.placeholder(tf.float32, [time_steps, batch_size, 1])

        # [time_steps, batch_size, num_inputs] -> [batch_size, num_inputs, batch_size, num_inputs]
        inputs_unstacked = tf.unstack(self.inputs, axis=0)
        labels_unstacked = tf.unstack(self.labels, axis=0)

        output = tf.constant([[0.0]])
        state = tf.constant([[0.0]])

        for input, label in zip(inputs_unstacked, labels_unstacked):
            output, state = self.lstm(input, output, state)
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