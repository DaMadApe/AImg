import numpy as np
import matplotlib.pyplot as plt
from skimage import data,transform

img = data.shepp_logan_phantom()

suma = np.zeros(img.shape)
for theta in range(0,180):
    captura = np.sum(transform.rotate(img, theta), axis=0)
    proyec = captura * np.ones((img.shape[1], 1))
    suma += transform.rotate(proyec, -theta)

plt.imshow(suma, cmap='gray')
plt.show()