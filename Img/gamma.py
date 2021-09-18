"""
Demostración de corrección gamma
"""
from skimage import data, io, color, transform
import imglib as lib
import numpy as np
import matplotlib.pyplot as plt

def gamma(img, g):
    scale = max(img.flatten())
    return img**g/scale**(g-1)

img = data.astronaut()
imgs = [img]
imgs.append(lib.trans_v(img, lambda x: gamma(x, 2)))
imgs.append(lib.trans_v(img, lambda x: gamma(x, 0.5)))
imgs.append(lib.trans_v(img, lambda x: gamma(x, 0.25)))

fig, ax = plt.subplots(len(imgs), 2, figsize=(8, 4*len(imgs)))

for i, im in enumerate(imgs):
    ax[i,0].imshow(im, cmap='gray')
    ax[i,0].axis('off')
    ax[i,1].plot(lib.histograma(color.rgb2gray(im)))

plt.tight_layout()
plt.show()