from skimage import data, io, color
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from histEq import histograma

img = np.arange(0, 255)

oscuros = fuzz.zmf(img, 25, 130)
grises = fuzz.gbellmf(img, 50, 3, 130)
claros = fuzz.smf(img, 130, 230)

# Singletons de conjuntos
s1 = 10
s2 = 130
s3 = 240

ecufuzz = np.zeros(256)
for i in range(255):
    ecufuzz[i] = (s1*oscuros[i] + s2*grises[i] + s3*claros[i])/(oscuros[i] + grises[i] + claros[i])


path = 'datos/pulmones2.jpeg'
img = io.imread(path)#data.astronaut()
img = color.rgb2gray(img)*255.0

[filas, columnas] = img.shape

img_eq = np.zeros( (filas, columnas) )

for i in range(filas):
    for j in range(columnas):
        valor = int(img[i, j])
        img_eq[i,j] = np.uint8(ecufuzz[valor])

h_in, bin_edges_in = np.histogram(img, bins=256,range = (0, 255))
h_out, bin_edges_out = np.histogram(img_eq, bins=256, range = (0, 255))


fig1, ax1 = plt.subplots(2)
ax1[0].plot(claros)
ax1[0].plot(grises)
ax1[0].plot(oscuros)
ax1[0].legend(['Claros', 'Grises', 'Oscuros'])
ax1[1].plot(ecufuzz)

fig2, ax2 = plt.subplots(2,2)
ax2[0, 0].imshow(img, cmap='gray')
ax2[0, 0].axis('off')
ax2[0, 1].plot(bin_edges_in[0:-1], h_in)
ax2[1, 0].imshow(img_eq, cmap='gray')
ax2[1, 0].axis('off')
ax2[1, 1].plot(bin_edges_out[0:-1], h_out)

plt.tight_layout()
plt.show()