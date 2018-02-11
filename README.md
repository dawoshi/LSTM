# Long short-term memory

## Introduction
This project shows how the BasicLSTMCell is implemented internally in Tensorflow.

## BasicLSTMCell
The implementation of Tensorflow's BasicLSTMCell is based on:
http://arxiv.org/abs/1409.2329

The forget gate 
```Python
forget_gate = tf.sigmoid(tf.matmul(input, w_f) + tf.matmul(u_f, hidden_state) + b_f)
```
outputs a number between 0 and 1. It decides how much of the old state we forget.

The input gate
```Python
input_gate = tf.sigmoid(tf.matmul(input, w_i) + tf.matmul(u_i, hidden_state) + b_i)
```
decides how much new input is part of the new state.

Then we create the information that could be added to the state:
```Python
new_input = tf.tanh(tf.matmul(input, w_j) + tf.matmul(u_j, hidden_state) + b_j)
```

We update the old cell state into the new cell state by using the gates:
```Python
new_state = new_input * input_gate + forget_gate * state
```

The output gate 
```Python
output_gate = tf.sigmoid(tf.matmul(input, w_o) + tf.matmul(u_i, hidden_state) + b_i)
```

decides which parts we output:
```Python
new_hidden_state = tf.tanh(new_state) * output_gate
```