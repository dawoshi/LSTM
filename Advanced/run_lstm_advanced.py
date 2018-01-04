"""
Using the BasicLSTMCell to predict the next symbol of a sequence.
complete sequence = [0, 0, 0, 1, 1, 1, 0]

after 0, 0, 0 follows 1
after 0, 0, 1 follows 1
after 0, 1, 1 follows 1
after 1, 1, 1 follows 0
"""

import random
import numpy as np
from Advanced.lstm_advanced import *

def run():
    random.seed(0)

    data = [
        ([[[0]], [[0]], [[0]]], [1]),
        ([[[0]], [[0]], [[1]]], [1]),
        ([[[0]], [[1]], [[1]]], [1]),
        ([[[1]], [[1]], [[1]]], [0]),
    ]



    model = Model()

    iterations = 500
    for i in range(0, iterations):
        random_index = random.randint(0, 3)
        inputs = np.asarray(data[random_index][0])
        label = np.asarray(data[random_index][1])
        loss = model.train(inputs, label)

        if i % 50 == 0:
            prediction = model.predict(inputs)
            print("***** {} *****".format(i))
            print("next symbol (predicted)", prediction)
            print("next symbol", label[0])
            print("loss", loss)

run()
