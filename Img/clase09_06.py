from operator import pos
from skimage import io, color
from skimage.morphology import binary_dilation
import numpy as np
import matplotlib.pyplot as plt

path = "datos/paciente/23.JPG"
img = io.imread(path)
gris = color.rgb2gray(img)

plt.figure(1)
plt.imshow(img, cmap='gray')
plt.ion() # Activar interactividad

tol = 15
# Semilla
pos = np.int32(plt.ginput(0, 0))

phi = np.zeros(img.shape, dtype = np.byte)
phi_old = np.zeros(img.shape, dtype = np.byte)

phi[pos[:, 1], pos[:, 0]] = 1
pixeles = gris[pos[:, 1], pos[:, 0]]
prom = np.mean(pixeles)

while np.sum(phi_old) != np.sum(phi):
    plt.cla() #limpiar fig
    phi_old = np.copy(phi)
    bordes = binary_dilation(phi) - phi
    pos_new = np.argwhere(bordes)
    pix_new = gris[pos_new[:, 0], pos_new[:, 1]]
    compara = list(np.logical_and((pix_new > prom-tol), (pix_new < prom+tol)))
    datos = pos_new[compara]
    phi[datos[:,0], datos[:,1]] = 1
    plt.imshow(phi, cmap='gray')
    plt.pause(0.01)

plt.figure(2)
plt.imshow(gris, cmap='gray')