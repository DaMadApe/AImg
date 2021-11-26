"""
Problema 2
Compuerta XOR con red de base radial
Daniel Sapién Garza
"""
import numpy as np

from PSO import Enjambre


class RedRBF():
    """
    Red de una capa intermedia de tamaño mid_size
    """
    def __init__(self, input_size, mid_size):
        self.W1 = np.random.rand(input_size, mid_size)
        self.b1 = np.random.rand(mid_size, 1)
        self.W2 = np.random.rand(mid_size, mid_size)
        self.b2 = np.random.rand(mid_size, 1)

    def forward(self, x):
        out = np.abs(self.W1 - x).T
        out = np.linalg.norm(out, axis=-1).reshape(-1,1)
        out = out*self.b1
        out = self._rbf(out)
        out = self.W2.dot(out) + self.b2
        return np.sum(out)

    def _rbf(self, x):
        return np.exp(-np.array(x)**2)


xor = [([[0], [0]], 0),
       ([[0], [1]], 1),
       ([[1], [0]], 1),
       ([[1], [1]], 0)]

red = RedRBF(2, 10)
print(red.forward([[0.5], [0.5]]))

def forward(self, params, x):
    W1 = np.reshape(params[0:mid_size*input_size], (input_size, mid_size))
    out = np.abs(W1 - x).T
    out = np.linalg.norm(out, axis=-1).reshape(-1,1)
    out = out*b1
    out = self._rbf(out)
    out = W2.dot(out) + b2
    return np.sum(out)