""" LSTM without using as BasicLSTMCell """

import tensorflow as tf

w_input_gate = tf.get_variable("w_input_gate", dtype=tf.float32, initializer=tf.constant([[0.26770878], [0.16705453]]))
w_new_input = tf.get_variable("w_new_input", dtype=tf.float32, initializer=tf.constant([[0.59700513], [0.16668344]]))
w_forget_gate = tf.get_variable("w_forget_gate", dtype=tf.float32, initializer=tf.constant([[0.33730936], [-0.220312]]))
w_output_gate = tf.get_variable("w_output_gate", dtype=tf.float32, initializer=tf.constant([[-0.92146528], [-0.4016704]]))

u_input_gate = tf.get_variable("u_input_gate", dtype=tf.float32, initializer=tf.constant([0.73747325]))
u_new_input = tf.get_variable("u_new_input", dtype=tf.float32, initializer=tf.constant([-0.81513613]))
u_forget_gate = tf.get_variable("u_forget_gate", dtype=tf.float32, initializer=tf.constant([0.33634782]))
u_output_gate = tf.get_variable("u_output_gate", dtype=tf.float32, initializer=tf.constant([0.90510082]))

b_input_gate = tf.get_variable("b_a", dtype=tf.float32, initializer=tf.constant([0.0]))
b_new_input = tf.get_variable("b_i", dtype=tf.float32, initializer=tf.constant([0.0]))
b_forget_gate  = tf.get_variable("b_f", dtype=tf.float32, initializer=tf.constant([0.0]))
b_output_gate = tf.get_variable("b_0", dtype=tf.float32, initializer=tf.constant([0.0]))

class Model:
    def __init__(self):
        tf.set_random_seed(0)
        self.loss = 0
        self.inputs = None
        self.labels = None
        self.optimizer = None
        self.session = tf.Session()
        self.build_graph()

    def lstm(self, input, state, output):
        input_gate = tf.sigmoid(tf.matmul(input, w_input_gate) + tf.multiply(u_input_gate, output) + b_input_gate)
        new_input = tf.tanh(tf.matmul(input, w_new_input) + tf.multiply(u_new_input, output) + b_new_input)
        forget_gate = tf.sigmoid(tf.matmul(input, w_forget_gate) + tf.multiply(u_forget_gate, output) + b_forget_gate)
        output_gate = tf.sigmoid(tf.matmul(input, w_output_gate) + tf.multiply(u_output_gate, output) + b_output_gate)
        state = new_input * input_gate + forget_gate * state
        output = tf.tanh(state) * output_gate
        return output, state

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

        state = 0
        output = 0

        for input, label in zip(inputs_unstacked, labels_unstacked):
            output, state = self.lstm(input, state, output)
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