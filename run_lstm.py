import random
import numpy as np
from model import Model
import rnn
import argparse

def run_lstm(implementation):
    random.seed(0)
    data = [
        ([[[0]], [[0]], [[0]]], [1]),
        ([[[0]], [[0]], [[1]]], [1]),
        ([[[0]], [[1]], [[1]]], [1]),
        ([[[1]], [[1]], [[1]]], [0]),
    ]

    model = Model(implementation)

    timesteps = 500
    for t in range(0, timesteps):
        random_index = random.randint(0, 3)
        inputs = np.asarray(data[random_index][0])

        label = np.asarray(data[random_index][1])
        loss = model.train(inputs, label)

        if t % 50 == 0:
            prediction = model.predict(inputs)
            print("***** {} *****".format(t))
            print("next symbol (predicted)", prediction)
            print("next symbol", label[0])
            print("loss", loss)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--implementation', help='BasicLSTMCell implementation', default='simple')
    args = parser.parse_args()

    if args.implementation == "simple":
        run_lstm(rnn.BasicLSTMCell)
    elif args.implementation == "tensorflow":
        run_lstm(rnn.tf.contrib.rnn.BasicLSTMCell)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
