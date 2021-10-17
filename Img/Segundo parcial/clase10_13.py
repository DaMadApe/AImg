from skimage import (io, data, color, exposure,
                     morphology, feature, util)
import numpy as np
import matplotlib.pyplot as plt

img = data.retina()

img_r = img[:,:,0]
img_g = img[:,:,1]
img_b = img[:,:,2]

img_ginv = util.invert(img_g)
img_ginv_eq = exposure.equalize_adapthist(img_ginv)
fondo = morphology.opening(img_ginv_eq, morphology.disk(10))
venas1 = img_ginv_eq - fondo


imgs = [(img, 'Imagen original'),
        #(img_r, 'Canal R'),
        (img_g, 'Canal G'),
        #(img_b, 'Canal B'),
        (img_ginv_eq, 'Canal G inv eq'),
        (venas1, 'Venas')]

for i, (img, title) in enumerate(imgs):
    plt.subplot(2, len(imgs)//2, i+1)
    if img.shape[-1] == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()