"""
Comparar funciones de histEq con funciones
de librería
"""
import matplotlib.pyplot as plt
import skimage as ski
from skimage import io, data, color, exposure
from imglib import histograma, acumulado, ecualizar

plt.close('all')

img = data.camera()
img_eq1 = ecualizar(img)
img_eq2 = exposure.equalize_hist(img)

images = [(img, 'original'),
          (img_eq1, 'ecualizada 1'),
          (img_eq2, 'ecualizada 2')]

n = len(images)
fig, ax = plt.subplots(n, 3, figsize=(9, 3*n))
for i, (im, tag) in enumerate(images):
    ax[i, 0].imshow(im, cmap='gray')
    ax[i, 0].axis('off')
    ax[i, 0].set_title(f'Imagen {tag}')

    ax[i, 1].plot(histograma(im))
    ax[i, 1].set_title(f'Histograma {tag}')

    hist, h_centers = exposure.histogram(im)
    ax[i, 2].plot(h_centers, hist)
    ax[i, 2].set_title(f'Histograma(SK) {tag}')

plt.tight_layout()
plt.show()