import numpy as NP
import random
import matplotlib.pyplot as plt
#------------------------------------------------------------------------
gamma = 0.8

recompensa = NP.array([[-1, -1, -1, -1, 0, -1],
                       [-1, -1, -1, 0, -1, 100],
                       [-1, -1, -1, 0, -1, -1],
                       [-1, 0, 0, -1, 0, -1],
                       [ 0, -1, -1, 0, -1, 100],
                       [-1, 0, -1, -1, 0, 100]])

qmatrix = NP.zeros(recompensa.shape)

tran_matrix = NP.array([[-1, -1, -1, -1, 4, -1],
                        [-1, -1, -1, 3, -1, 5],
                        [-1, -1, -1, 3, -1, -1],
                        [-1, 1, 2, -1, 4, -1],
                        [ 0, -1, -1, 3, -1, 5],
                        [-1, 1, -1, -1, 4, 5]])

accion_matrix = [[4], [3, 5], [3], [1, 2, 4], [0, 3, 5], [1, 4, 5]]

secuencia = []
for itera in range(100):
    
    estado_inicio = random.choice( list(range(0,recompensa.shape[0])))
    estado_actual = estado_inicio
    secuencia.append( estado_inicio )
    
    while estado_actual != 5:
        
        accion = random.choice( accion_matrix[estado_actual] )
        estado_siguiente = tran_matrix[estado_actual][accion]
        recompensa_siguiente = []
        
        for accion_siguiente in accion_matrix[estado_siguiente]:
            recompensa_siguiente.append(qmatrix[estado_siguiente][accion_siguiente])
            
        qestado = recompensa[estado_actual][accion] + gamma*max(recompensa_siguiente)
        qmatrix[estado_actual][accion] = qestado
        print(qmatrix)
        estado_actual = estado_siguiente
        
        if estado_actual == 5:
            print('Meta alcanzada')

print('Matriz Q final = ')
print(qmatrix)
