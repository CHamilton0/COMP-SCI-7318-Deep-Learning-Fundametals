from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd


class RNN(ABC):
    input_size: int
    hidden_size: int

    W_xh: npt.NDArray[np.float64]
    W_hh: npt.NDArray[np.float64]
    W_hy: npt.NDArray[np.float64]
    h: npt.NDArray[np.float64]

    @abstractmethod
    def step(self, x: npt.NDArray[np.float64]) -> float:
        pass

    @abstractmethod
    def forward(self, inputs: list[npt.NDArray[np.float64]]) -> float:
        pass

    @abstractmethod
    def backward(
        self,
        inputs: list[pd.Series],
        target: pd.Series,
        output: float,
        learning_rate: float = 0.01,
    ) -> None:
        pass
