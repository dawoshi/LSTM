# Long short-term memory

## Introduction
This project shows how the BasicLSTMCell is implemented internally in Tensorflow.

## BasicLSTMCell
The implementation of Tensorflow's BasicLSTMCell is based on:
http://arxiv.org/abs/1409.2329

The LSTM architecture is defined by the following equations:

```Python
forget_gate =   sigmoid(matmul(input, w_f) + matmul(u_f, hidden_state) + b_f)
input_gate =    sigmoid(matmul(input, w_i) + matmul(u_i, hidden_state) + b_i)
new_input =     tanh(matmul(input, w_j)    + matmul(u_j, hidden_state) + b_j)
output_gate =   sigmoid(matmul(input, w_o) + matmul(u_o, hidden_state) + b_o)

new_cell_state = cell_state * forget_gate + input_gate * new_input
new_hidden_state = tanh(new_cell_state) * output_gate
```

* The forget gate outputs a number between 0 and 1. It decides how much of the old cell state we forget.
* The input gate decides how much new input is part of the new state.
* The output gate decides which parts we output.

## Example
In this project the BasicLSTMCell is used to predict the next symbol of the sequence
```
[0, 0, 0, 1, 1, 1, 0]. 
```

The sequence is divided into inputs of 3 time steps:

| Input | Label |
| --- | ---
| [0, 0, 0] | 1 |
| [0, 0, 1] | 1 |
| [0, 1, 1] | 1 |
| [1, 1, 1] | 0 |

## Run LSTM
```shell
$ python -m run_lstm
```

