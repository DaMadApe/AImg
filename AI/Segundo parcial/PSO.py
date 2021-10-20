import numpy as np
import matplotlib.pyplot as plt


class Particula:

    def __init__(self, dims, min_val, max_val):
        self.pos = np.random.uniform(min_val, max_val, dims)
        self.vel = np.random.uniform(min_val-max_val,
                                     max_val-min_val, dims)
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
        self.particulas = []
        for _ in range(n_particulas):
            self.particulas.append(Particula(dims, min_val, max_val))
        self.bestCost = np.inf
        self.bestPos = None #np.random.choice(self.particulas).bestPos
        self.costHist = []

    def minimizar(self, cost_fun, c1, c2, w, iters):
        # Inicializar costos
        for p in self.particulas:
            costo = cost_fun(p.bestPos)
            p.newBest(costo)
            if costo < self.bestCost:
                self.bestCost = costo
        # Entrenamiento
        for _ in range(iters):
            for p in self.particulas:
                # Actualizar pos y vel de partícula
                r1, r2 = np.random.uniform(size=2)
                deltaV = w*p.vel + c1*r1*(p.bestPos-p.pos) + c2*r2*(self.bestPos-p.pos) 
                p.paso(deltaV)

                cost = cost_fun(p.pos)
                # Actualizar mejores valores de partícula
                if cost < p.bestCost:
                    p.newBest(cost)
                # Actualizar mejores valores globales
                if cost < self.bestCost:
                    self.bestCost = cost
                    self.bestPos = p.pos.copy()
            if self.bestCost < np.inf:
                self.costHist.append(self.bestCost)

    # Tal vez poner min,max en metodo minimizar, y normalizar pos, vel

    def minimizar_local(self, cost_fun, ):
        pass

    def minimizar_social(self, cost_fun, n_grupos, iters):
        """
        Repartir índices en n_grupos
        ABCABCABC
        AAABBBCCC
        Instanciar swarm para cada grupo?
        """
        pass


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

    swarm = Enjambre(100, 3, -100, 100)
    swarm.minimizar(sphere, c1, c2, w, 1000)

    print(swarm.bestPos, swarm.bestCost)

    plt.plot(swarm.costHist)
    plt.show()