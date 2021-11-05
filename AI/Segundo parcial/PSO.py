import numpy as np
import matplotlib.pyplot as plt
import nevergrad as ng
from nevergrad.optimization.optimizerlib import ConfiguredPSO


class Particula:

    def __init__(self, dims, min_val, max_val):
        self.pos = np.random.uniform(min_val, max_val, dims)
        self.vel = np.random.uniform(min_val - max_val,
                                     max_val - min_val, dims)
        self.bestPos = self.pos.copy()
        self.bestCost = None #np.inf

    def paso(self, deltaVel):
        self.vel += deltaVel
        self.pos += self.vel

    def newBest(self, newCost):
        self.bestPos = self.pos.copy()
        self.bestCost = newCost


class Enjambre:

    def __init__(self, n_particulas, dims, min_val, max_val):
        self.n_particulas = n_particulas
        self.dims = dims
        self.min_val = min_val
        self.max_val = max_val
        self.particulas = []
        for _ in range(n_particulas):
            self.particulas.append(Particula(dims, min_val, max_val))
        self.bestCost = 100000 #np.inf
        self.bestPos = None #np.random.choice(self.particulas).bestPos
        self.costHist = []

    def minimizar(self, cost_fun, c1, c2, w, iters):
        # Inicializar costos
        for p in self.particulas:
            costo = cost_fun(p.bestPos)
            p.newBest(costo)
            if costo < self.bestCost:
                self.bestCost = costo
                self.bestPos = p.pos
        # Entrenamiento
        for _ in range(iters):
            for p in self.particulas:
                # Actualizar pos y vel de partícula
                r1, r2 = np.random.uniform(size=2)
                deltaV = (w*p.vel
                          + c1*r1*(p.bestPos - p.pos)
                          + c2*r2*(self.bestPos - p.pos)) 
                p.paso(deltaV)

                cost = cost_fun(p.pos)
                # Actualizar mejores valores de partícula
                if cost < p.bestCost:
                    p.newBest(cost)
                # Actualizar mejores valores globales
                if cost < self.bestCost:
                    self.bestCost = cost
                    self.bestPos = p.pos.copy()
            # Registrar pérdidas para graficar su curva
            if self.bestCost < np.inf:
                self.costHist.append(self.bestCost)

    # Tal vez poner min,max en metodo minimizar, y normalizar pos, vel

    # def minimizar_social(self, cost_fun, n_grupos, c1, c2, w, iters):
    #     """
    #     Repartir índices en n_grupos
    #     ABCABCABC
    #     AAABBBCCC
    #     Instanciar swarm para cada grupo?
    #     """
    #     N = self.n_particulas//n_grupos
    #     for i in range(n_grupos):
    #         grupo = Enjambre(N, self.dims, 
    #                          self.min_val, self.max_val)
    #         grupo.particulas = self.particulas[N*i:N*(i+1)]
    #         grupo.minimizar(cost_fun, c1, c2, w, iters)
    #         self.particulas[N*i:N*(i+1)] = grupo.particulas
    #     for p in self.particulas:
    #         if p.bestCost < self.bestCost:
    #             self.bestCost = p.bestCost
    #             self.bestPos = p.bestPos