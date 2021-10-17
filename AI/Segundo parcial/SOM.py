"""
Mapa autoorganizado para producir una representación
bidimensional de un espacio de color.

Daniel Sapién Garza
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import dtype


class SOM():

    def __init__(self, in_size, out_shape):
        self.in_size = in_size
        self.out_shape = out_shape
        # Inicializar pesos aleatoriamente
        self.w = np.random.rand(*out_shape, in_size)
        self.w_init = self.w.copy()

    def __call__(self, p):
        # Op 1: Producto punto con matriz de pesos original
        # return np.argmax(self.w.dot(p))

        # Op 2: Producto punto con matriz de pesos normalizada
        # w = self.w / np.linalg.norm(self.w, axis=-1, keepdims=True)
        # return np.argmax(w.dot(p))

        # Op 3: Distancia euclidiana
        return np.argmin(np.sum((self.w - p)**2, axis=-1))

    def train(self, trainset, alfa, r, epochs, r_decay=False):
        for i in range(epochs):
            for p in trainset:
                # Normalizar vectores prototipo, luminancia constante
                self.w /= np.linalg.norm(self.w, axis=-1, keepdims=True)

                # Activación de modelo (llamada de __call__)
                i = self(p)
                # Determinar neuronas en radio de influencia
                mask = self._vecinos(i, r)
                # Actualización de pesos
                self.w[mask] += alfa*(p - self.w[mask])
            # Decaimiento del ritmo de aprendizaje
            alfa *= 0.8
            # Decaimiento del radio de influencia
            if r_decay:
                r = r-1

    def _vecinos(self, idx, r):
        # Encontrar los índices de neuronas dentro de un radio
        y0, x0 = np.unravel_index(idx, self.out_shape)
        n, m = self.out_shape 
        y, x = np.ogrid[-y0:n-y0, -x0:m-x0]
        mask = x**2 + y**2 <= r**2
        return mask


np.random.seed(42)

# Parámetros
alfa = 0.1 # Learning rate
r = 20 # Radio de influencia
N = 1000 # Número de muestras
epochs = r-1 # Recorridos del conjunto de entrenamiento

# Datos de entrenamiento
train_set = np.random.rand(N, 3)

# Instanciar y entrenar modelo
som = SOM(in_size=3, out_shape=(32, 32))
som.train(train_set, alfa, r, epochs, r_decay=True)


# Visualizar pesos antes y después de entrenamiento
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(som.w_init)
plt.title('Pesos W iniciales')
plt.subplot(1,2,2)
plt.imshow(som.w)
plt.title('Pesos organizados')

# Registro de experimentos
# plt.savefig(f'colorSOM/ini_a{alfa}_r{r}_e{epochs}_N{N}.png') # # #
plt.show()