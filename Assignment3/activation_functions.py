import math


def leaky_relu(x: float) -> float:
    return max(0.1 * x, x)


def relu(x: float) -> float:
    return max(0, x)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-1 * x))


def tanh(x: float) -> float:
    return math.tanh(x)


def elu(x: float, a: float = 0.5) -> float:
    return x if x > 0 else a * (math.exp(x) - 1)
