import numpy as np
import nevergrad as ng
import matplotlib.pyplot as plt

from PSO import Enjambre


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

swarm = Enjambre(100, 3, -1, 1)
swarm.minimizar(sphere, c1, c2, w, 1000)

ng_pso = ng.optimizers.ConfiguredPSO(omega=w, phip=c1, phig=c2)
ng_result = ng_pso.minimize(sphere)

print("Algoritmo propio")
print(swarm.bestPos, swarm.bestCost)
print("Algoritmo de Nevergrad")
print(ng_result.value)

# Graficar curva de pérdidas
# plt.plot(swarm.costHist)
# plt.show()