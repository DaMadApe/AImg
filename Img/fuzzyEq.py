from skimage import data, io, color
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzzy
from imglib import ecu_hist, histograma, ecu_fuzzy

path = 'datos/pulmones2.jpeg'
img = io.imread(path)#data.astronaut()
img = color.rgb2gray(img)*255
img = np.rint(img)

# Singletons de conjuntos
s1 = 10
s2 = 130
s3 = 240
singl = [s1, s2, s3]

# Conjuntos difusos
tone_range = np.arange(255)
oscuros = fuzzy.zmf(tone_range, 25, 130)
grises = fuzzy.gbellmf(tone_range, 50, 3, 130)
claros = fuzzy.smf(tone_range, 130, 230)

fuzzy_sets = np.stack([oscuros, grises, claros])

#Ecufuzz para graficar
ecufuzz = np.zeros(256)
for i in range(255):
    ecufuzz[i] = np.dot(fuzzy_sets[:, i], singl)/sum(fuzzy_sets[:,i])

img_eq1 = ecu_hist(img)
img_eq2 = ecu_fuzzy(img, singl, fuzzy_sets)

h_in, bin_edges_in = np.histogram(img, bins=256,range = (0, 255))
h_out1, bin_edges_out1 = np.histogram(img_eq1, bins=256, range = (0, 255))
h_out2, bin_edges_out2 = np.histogram(img_eq2, bins=256, range = (0, 255))

fig1, ax1 = plt.subplots(2)
ax1[0].plot(claros)
ax1[0].plot(grises)
ax1[0].plot(oscuros)
ax1[0].legend(['Claros', 'Grises', 'Oscuros'])
ax1[1].plot(ecufuzz)

plt.tight_layout()

fig2, ax2 = plt.subplots(3,2)
ax2[0, 0].imshow(img, cmap='gray')
ax2[0, 0].axis('off')
ax2[0, 1].plot(histograma(img))#h_in)
ax2[1, 0].imshow(img_eq1, cmap='gray')
ax2[1, 0].axis('off')
ax2[1, 1].plot(h_out1)
ax2[2, 0].imshow(img_eq2, cmap='gray')
ax2[2, 0].axis('off')
ax2[2, 1].plot(h_out2)

plt.tight_layout()

plt.show()