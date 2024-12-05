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
        # Step through each input and calculate hidden states
        for x in inputs:
            y = self.step(x)
            self.hs.append(self.h)
        return y

    def backward(
        self,
        inputs: list[pd.Series],
        target: float,
        output: float,
        learning_rate: float = 0.01,
    ) -> None:
        # Initialize gradients
        gradient_W_xh = np.zeros_like(self.W_xh)
        gradient_W_hh = np.zeros_like(self.W_hh)
        gradient_W_hy = np.zeros_like(self.W_hy)

        # Derivative of the loss w.r.t. the output using MSE loss
        dL_dy = -2 * (target - output)

        dL_dh_next = np.zeros_like(self.hs[0])  # Initialize hidden gradient

        for t in reversed(range(len(inputs))):  # Propoagate backwards through time
            # Gradient w.r.t. weights from hidden state to output layer
            gradient_W_hy += dL_dy * self.hs[t].T

            # Backpropagate into hidden state
            dL_dh_sum = self.W_hy.T * dL_dy + dL_dh_next  # Combine gradients

            # Gradient w.r.t. hidden state (calculated with tanh)
            dL_dh = (1 - self.hs[t] ** 2) * dL_dh_sum

            # Calculate gradients for W_xh and W_hh
            gradient_W_xh += np.dot(dL_dh, inputs[t].T)
            gradient_W_hh += np.dot(dL_dh, self.hs[t - 1].T if t > 0 else 0)

            # Pass the gradient to next step in backwards propagation
            dL_dh_next = np.dot(self.W_hh.T, dL_dh)

        # Update weights
        self.W_xh -= learning_rate * gradient_W_xh
        self.W_hh -= learning_rate * gradient_W_hh
        self.W_hy -= learning_rate * gradient_W_hy
