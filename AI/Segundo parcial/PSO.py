import numpy as np
import matplotlib.pyplot as plt


class Particula:

    def __init__(self, dims, min_val, max_val):
        self.pos = np.random.uniform(min_val, max_val, dims)
        self.vel = np.random.uniform(min_val-max_val,
                                     max_val-min_val, dims)
        self.costo
        self.bestPos = self.pos.copy()
        self.bestCost = np.inf

    def paso(self, deltaVel):
        self.vel += deltaVel
        self.pos += self.vel

    def newBest(self, newCost):
        self.bestPos = self.pos.copy()
        self.bestCost = newCost

class Enjambre:

    def __init__(self, n_particulas, dims, min_val, max_val):
        self.particulas = []
        for _ in range(n_particulas):
            self.particulas.append(Particula(dims, min_val, max_val))

    def minimizar(self, cost_fun, c_c, c_s, w, iters):
        for i, p in enumerate(self.particulas):
            for 

    def minimizar_local(self, cost_fun, ):
        pass

    def minimizar_social(self, cost_fun, n_grupos, iters):
        """
        En luga
        """