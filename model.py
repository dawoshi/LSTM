import tensorflow as tf
import rnn

class Model:
    def __init__(self, basic_lstm_cell):
        tf.set_random_seed(0)
        self.session = tf.Session()
        timesteps = 3
        batch_size = 1
        num_inputs = 1
        num_units = 2  # without another layer, lstm units must match output
        learning_rate = 0.1

        self.inputs = tf.placeholder(tf.float32, [timesteps, batch_size, num_inputs])
        self.label = tf.placeholder(tf.uint8, [batch_size])
        """
        unstack:
        (3, 1, 1) -> [(1,1), (1,1), (1,1)]
        """
        inputs_unstacked = tf.unstack(self.inputs, axis=0)

        lstm_cell = basic_lstm_cell(num_units=num_units, forget_bias=0)

        if basic_lstm_cell == tf.contrib.rnn.BasicLSTMCell:
            self.outputs, _ = tf.nn.static_rnn(cell=lstm_cell,
                                               inputs=inputs_unstacked,
                                               dtype=tf.float32)
        elif basic_lstm_cell == rnn.BasicLSTMCell:
            state_tuple = lstm_cell.zero_state(batch_size)
            self.outputs = []
            for input in inputs_unstacked:
                hidden, state_tuple = lstm_cell(input, state_tuple)
                self.outputs.append(hidden)
        else:
            NotImplementedError

        """
        outputs = [(1,2), (1,2), (1,2)]
        """

        mask = tf.one_hot(self.label, depth=2)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.outputs[-1],
            labels=mask))

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def predict(self, inputs):
        prediction = self.session.run(tf.argmax(self.outputs[-1],1), feed_dict={self.inputs: inputs})
        return prediction[0]

    def train(self, inputs, labels):
        feed_dict = {
            self.inputs: inputs,
            self.label: labels
        }
        loss, _ = self.session.run([self.loss, self.optimizer],
                                   feed_dict=feed_dict)
        return loss