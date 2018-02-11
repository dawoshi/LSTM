import tensorflow as tf

class Model:
    def __init__(self):
        tf.set_random_seed(0)
        self.outputs = 0
        self.states = 0
        self.loss = 0
        self.inputs = None
        self.labels = None
        self.optimizer = None
        self.session = tf.Session()
        self.build_graph()

    def build_graph(self):
        time_steps = 3
        batch_size = 1
        num_inputs = 1
        num_units = 2 # without another layer, lstm units must match output
        learning_rate = 0.1

        self.inputs = tf.placeholder(tf.float32, [time_steps, batch_size, num_inputs])
        self.label = tf.placeholder(tf.uint8, [batch_size])

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units, 0)
        inputs_unstacked = tf.unstack(self.inputs, axis=0) # (3, 1, 1) -> [(1,1), (1,1), (1,1)]

        self.outputs, self.states = tf.nn.static_rnn(lstm_cell, inputs_unstacked, dtype=tf.float32) # [(1,2), (1,2), (1,2)]

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs[-1],
                                                            labels=tf.one_hot(self.label, depth=2))

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def predict(self, inputs):
        prediction = self.session.run(tf.argmax(self.outputs[-1],1), feed_dict={self.inputs: inputs})
        return prediction[0]

    def train(self, inputs, labels):
        loss, _ = self.session.run([self.loss, self.optimizer],
                                                 feed_dict={
                                                     self.inputs: inputs,
                                                     self.label: labels
                                                 })
        return loss[0]