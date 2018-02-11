import tensorflow as tf
import tensorflow.contrib.eager as tfe

def lstm(input, output, state, state_weights, hidden_state_weights):
    gate_inputs = tf.matmul(input, state_weights) + tf.matmul(output, hidden_state_weights)
    input_gate, new_input, forget_gate, output_gate = tf.split(axis=1, num_or_size_splits=4, value=gate_inputs)
    new_state = tf.tanh(new_input) * tf.sigmoid(input_gate) + tf.sigmoid(forget_gate) * state
    new_hidden_state = tf.tanh(new_state) * tf.sigmoid(output_gate)
    return new_hidden_state, new_state

# [array([[1.5335395]], dtype=float32)]

def test():
    tfe.enable_eager_execution()

    x0 = [[1.0, 2.0]]
    x1 = [[0.5, 3.0]]
    inputs = [x0, x1]

    label0 = [[0.5]]
    label1 = [[1.25]]
    labels = [label0, label1]

    output = tf.constant([[0.0]])
    state = 0

    state_weights = tf.get_variable("w", dtype=tf.float32, initializer=tf.constant(
        [
            [0.26770878, 0.59700513, 0.33730936, -0.92146528],
            [0.16705453, 0.16668344, -0.220312,-0.4016704]
        ]
    ))

    hidden_state_weights = tf.get_variable("u", dtype=tf.float32, initializer=tf.constant(
        [[0.73747325, -0.81513613, 0.33634782, 0.90510082]]
    ))

    loss = 0
    for input, label in zip(inputs, labels):
        output, state = lstm(input, output, state, state_weights, hidden_state_weights)
        loss += tf.square(output - label)

    print(loss)

test()

