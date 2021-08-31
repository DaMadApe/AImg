import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzzy
from skimage.exposure import histogram
from skimage.io import imread
from skimage.color import rgb2hsv

path = "datos/panda.jpg"
img = imread(path)
img_hsv = rgb2hsv(img)
img_v = img_hsv[:, :, 2]

# Histogramas por color y de valor
r_hist, r_centers = histogram(img[:,:,0])
g_hist, g_centers = histogram(img[:,:,1])
b_hist, b_centers = histogram(img[:,:,2])
hist, h_centers = histogram(img_hsv[:, :, 2])

fig1, ax1 = plt.subplots(2, 2, figsize=(10, 6))
ax1[0, 0].imshow(img)
ax1[0, 0].axis('off')
ax1[0, 0].set_title('Imagen original')

ax1[0, 1].plot(r_centers, r_hist, color='r', lw=1)
ax1[0, 1].plot(g_centers, g_hist, color='g', lw=1)
ax1[0, 1].plot(b_centers, b_hist, color='b', lw=1)
ax1[0, 1].set_title('Histogramas RGB')

ax1[1, 0].imshow(img_v, cmap='gray')
ax1[1, 0].axis('off')
ax1[1, 0].set_title('Imagen (canal V)')

ax1[1, 1].plot(h_centers, hist, color='k', lw=1)
ax1[1, 1].set_title('Histograma V')

plt.tight_layout()



# Fuzzy c means
c = 3
m = 2
data = np.reshape(img_v, (1, -1))
cntr, u, _, _, _, _, fpc = fuzzy.cmeans(data,
                                        c=c,
                                        m=m,
                                        error=5e-4,
                                        maxiter=16)

# Formar una imagen de cada cluster
clusters = []
for cluster in u:
    clusters.append(np.reshape(cluster, np.shape(img_v)))

# Función de membresía para graficarla
def mship(x, i):
    den = 0
    for k in range(c):
        den += abs(x-cntr[i])/abs(x-cntr[k])**(2/(m-1))
    return 1/den

# Membresía por conjunto
xs = np.linspace(0, 1, 100)
mships = [[mship(x, i) for x in xs] for i in range(c)]

fig2, ax2 = plt.subplots(2, c, figsize=(4*c, 6))
# Graficar cada cluster
for i, cluster in enumerate(clusters):
    ax2[0, i].imshow(cluster, cmap='gray')
    ax2[0, i].axis('off')
    ax2[0, i].set_title(f"Cluster {i}")

    ax2[1,i].plot(xs, mships[i], color='k')
    ax2[1,i].set_title(f"Membresía de cluster {i}")

plt.tight_layout()
plt.show()