from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from rnn import RNN


class SimpleRNN(RNN):
    activation_function: Callable[[float], float]

    def __init__(
        self,
        input_size: int,  # Length of input array
        hidden_size: int,  # Length of hidden states
        activation_function: Callable[[float], float],
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = (
            np.random.randn(hidden_size, input_size) * 0.01
        )  # Weights from input layer to hidden layer
        self.W_hh = (
            np.random.randn(hidden_size, hidden_size) * 0.01
        )  # Weights from hidden layer to hidden layer
        self.W_hy = (
            np.random.randn(1, hidden_size) * 0.01
        )  # Weights from hidden layer to output layer
        self.h = np.zeros((hidden_size, 1))  # Initialize hidden state

        self.activation_function = activation_function

    def step(self, x: npt.NDArray[np.float64]) -> float:
        x = x.reshape(-1, 1)  # Reshape to column vector
        self.h = np.tanh(
            np.dot(self.W_xh, x) + np.dot(self.W_hh, self.h)
        )  # Calculate the next hidden state
        y = self.activation_function(
            np.dot(self.W_hy, self.h)
        )  # Calculate the prediction for this input
        return float(np.asarray(y).item())

    def forward(self, inputs: list[npt.NDArray[np.float64]]) -> float:
        self.hs = []  # To store hidden states
        for x in inputs:
            y = self.step(x)
            self.hs.append(self.h)
        return y

    def backward(
        self,
        inputs: list[pd.Series],
        target: pd.Series,
        output: float,
        learning_rate: float = 0.01,
    ) -> None:
        # Compute gradient of the loss with respect to output
        d_loss = 2 * (output - target)  # Derivative of MSE

        # Backpropagate through the output layer
        d_W_hy = np.dot(d_loss, self.h.T)

        # Initialize hidden state gradients
        d_h = np.dot(self.W_hy.T, d_loss)
        d_W_hh = np.zeros_like(self.W_hh)
        d_W_xh = np.zeros_like(self.W_xh)

        # Backpropagate through time
        for t in reversed(range(len(inputs))):
            d_h_raw = (1 - self.hs[t] ** 2) * d_h  # Derivative of tanh
            d_W_hh += np.dot(
                d_h_raw, self.hs[t - 1].T if t > 0 else np.zeros_like(self.h).T
            )
            d_W_xh += np.dot(d_h_raw, inputs[t].T)
            d_h = np.dot(self.W_hh.T, d_h_raw)

        # Update weights
        self.W_hy -= learning_rate * d_W_hy
        self.W_hh -= learning_rate * d_W_hh
        self.W_xh -= learning_rate * d_W_xh
