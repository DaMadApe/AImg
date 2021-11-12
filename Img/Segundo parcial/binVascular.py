import numpy as np
import matplotlib.pyplot as plt
from skimage import (io, data, exposure, util,
                     filters, morphology)

img = io.imread('datos/ojo/03_L.jpg')
#img = data.retina()
img_g = img[:,:,1]

img_eq = util.img_as_ubyte(exposure.equalize_adapthist(img_g))
#img_eq2 = exposure.adjust_log(img_g, 1)

img_bin = img_eq > filters.threshold_otsu(img_eq)
img_bin2 = img_eq > filters.rank.otsu(img_eq, morphology.disk(15))


# Mostrar imágenes
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Imagen original')
plt.subplot(2,2,2)
plt.imshow(img_g, cmap='gray')
plt.title('Canal verde aislado')
plt.subplot(2,2,3)
plt.imshow(img_bin, cmap='gray')
plt.subplot(2,2,4)
plt.imshow(img_bin2, cmap='gray')

plt.show()