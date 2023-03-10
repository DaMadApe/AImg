import numpy as np
import matplotlib.pyplot as plt


class Particula:

    def __init__(self, dim, min_val, max_val):
        self.dim = dim
        self.min_val = min_val
        self.max_val = max_val
        self.pos = np.random.uniform(min_val, max_val, dim)
        self.vel = np.random.uniform(min_val - max_val,
                                     max_val - min_val, dim)
        self.bestPos = self.pos.copy()
        self.bestCost = None #np.inf

    def paso(self, nextVel):
        self.vel = nextVel
        self.pos += self.vel
        # Restringir a espacio de búsqueda
        self.pos = np.clip(self.pos, self.min_val, self.max_val)

    def newBest(self, newCost):
        self.bestPos = self.pos.copy()
        self.bestCost = newCost


class Enjambre:

    def __init__(self, n_particulas, dim, min_val, max_val):
        self.n_particulas = n_particulas
        self.dim = dim
        self.min_val = min_val
        self.max_val = max_val
        self.particulas = []
        for _ in range(n_particulas):
            self.particulas.append(Particula(dim, min_val, max_val))
        self.bestCost = 100000 #np.inf
        self.bestPos = None #np.random.choice(self.particulas).bestPos
        self.costHist = []

    def minimizar(self, cost_fun, c1, c2, w, iters):
        # Inicializar costos
        for p in self.particulas:
            costo = cost_fun(p.pos)
            p.newBest(costo)
            if costo < self.bestCost:
                self.bestCost = costo
                self.bestPos = p.pos
        # Entrenamiento
        for _ in range(iters):
            for p in self.particulas:
                # Actualizar pos y vel de partícula
                r1, r2 = np.random.uniform(size=2)
                nextV = (w*p.vel
                         + c1*r1*(p.bestPos - p.pos)
                         + c2*r2*(self.bestPos - p.pos)) 
                p.paso(nextV)

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


"""
Pruebas
"""
if __name__ == '__main__':

    def sphere(x):
        #min @ (0, 0) = 0
        return np.sum(x**2)

    def eggholder(x):
        #min @ (512, 404.2319) = -959.6407
        term1 = -(x[1]+47) * np.sin(np.sqrt(abs(x[0]/2 + x[1]+47)))
        term2 = -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]+47))))
        return term1 + term2

    c1 = 1.9
    c2 = 1.9
    w = 1
    iters = 500
    criterio = eggholder

    swarm = Enjambre(n_particulas=40, dim=2, min_val=0, max_val=512)
    swarm.minimizar(criterio, c1, c2, w, iters)

    print("(Mejor posición, mejor costo)")
    print(swarm.bestPos, swarm.bestCost)

    # Graficar curva de pérdidas
    min_cost = min(swarm.costHist)
    # Asegurar costos positivos para usar escala logarítmica
    costHist = swarm.costHist - min(0, min_cost)
    plt.plot(costHist)
    plt.yscale("log")
    plt.show()