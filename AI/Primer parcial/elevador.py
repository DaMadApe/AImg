"""
Problema Hagan E15.9
Daniel Sapién Garza
"""
import numpy as np

def hardlim(x):
    if x > 0:
        return 1
    return -1

def floor_vec(floor):
    """
    Devolver el vector correspondiente a un piso
    """
    floors = np.array([[ 0,  0],
                       [-1, -1],
                       [ 1, -1],
                       [-1,  1],
                       [ 1,  1]])
    return floors[floor].reshape(-1, 1)

def person_vec(person):
    """
    Devolver el vector correspondiente a una persona
    """
    vips = ['P', 'VP', 'C']
    vips = np.array(vips).reshape(-1,1)
    vec = vips == person
    return vec.astype(int)

def modelo(W, p0, p):
    W0 = np.eye(2) # Pesos para código de piso
    b = -0.5 * np.ones((2,1)) # Bias
    a = np.vectorize(hardlim)(W0.dot(p0) + W.dot(p) + b)
    return a

def train_outstar(W, modelo, train_set, lr, epochs=1):
    for t in range(epochs):
        for dp in train_set:
            p0, p = dp
            a = modelo(W, p0, p)
            W = W + lr * np.dot((a - W), p) * p.T
            for i in range(W.shape[0]):
                W[:,i] = W[:,i] + lr * (a.flatten() - W[:,i])*p[i, 0]
    return W



"""
I
"""
# Secuencia de entrenamiento
train_set_1 = [(floor_vec(4), person_vec('P')),
               (floor_vec(3), person_vec('VP')),
               (floor_vec(1), person_vec('C')),
               (floor_vec(3), person_vec('VP')),
               (floor_vec(2), person_vec('C')),
               (floor_vec(4), person_vec('P'))]

# Parámetros iniciales
W = np.zeros((2,3)) # Para código de persona
# Entrenamiento de red
W = train_outstar(W, modelo, train_set_1, lr=0.6)

"""
II
"""
print(W)

"""
III, IV
"""
# Probar predicciones cuando no se presiona botón
print("Predicción Presidente")
print(modelo(W, floor_vec(0), person_vec('P')))

print("Predicción VP")
print(modelo(W, floor_vec(0), person_vec('VP')))

print("Predicción C")
print(modelo(W, floor_vec(0), person_vec('C')))