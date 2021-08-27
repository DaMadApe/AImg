import matplotlib.pyplot as plt
import skimage as ski
from skimage.io import imread
from skimage.color import rgb2hsv, hsv2rgb
from histEq import histograma, acumulado, ecualizar

img = ski.data.retina()
img_hsv = rgb2hsv(img)
img_v = img_hsv[:, : ,2].copy()

img_v_eq = ecualizar(img_v)
img_hsv[:, : ,2] = img_v_eq

img_eq = hsv2rgb(img_hsv)

fig, ax = plt.subplots(2, 3, figsize=(16, 10))

ax[0, 0].imshow(img, cmap='gray')
ax[0, 0].axis('off')
ax[0, 0].set_title('Imagen original')

ax[0, 1].plot(histograma(img_v))
ax[0, 1].set_title('Histograma V original')

ax[1, 0].imshow(img_eq, cmap='gray')
ax[1, 0].axis('off')
ax[1, 0].set_title('Imagen ecualizada')

ax[1, 1].plot(histograma(img_v_eq))
ax[1, 1].set_title('Histograma V ecualizado')

ax[0, 2].plot(acumulado(histograma(img_v)))
ax[0, 2].set_title('Histograma V acumulado')

ax[1,2].axis('off')

plt.tight_layout()
plt.show()