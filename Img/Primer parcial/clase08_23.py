import skimage
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

img = skimage.data.astronaut()
img_hsv = rgb2hsv(img)

# pruebas con V
fig1, ax1 = plt.subplots(2, 2, figsize=(8, 8))
ax1[0, 0].imshow(img)
ax1[0, 0].axis('off')
ax1[0, 0].set_title('Imagen original')

v = img_hsv[:,:,2]
ax1[0, 1].imshow(v, cmap='gray')
ax1[0, 1].axis('off')
ax1[0, 1].set_title('Imagen canal V')

c_v=0.8
img_hsv[:,:,2] = v*c_v
img_dim = hsv2rgb(img_hsv)
ax1[1, 0].imshow(img_dim, cmap='gray')
ax1[1, 0].axis('off')
ax1[1, 0].set_title(f'Imagen HSV, V * {c_v}')

ax1[1, 1].imshow(v*c_v, cmap='gray',
                 norm=Normalize(0, 1))
ax1[1, 1].axis('off')
ax1[1, 1].set_title(f'Canal V * {c_v}')

plt.tight_layout()

# Pruebas con H
img_hsv = rgb2hsv(img)

fig2, ax2 = plt.subplots(2, figsize=(4, 8))
ax2[0].imshow(img)
ax2[0].axis('off')
ax2[0].set_title('Imagen original')

c_h = 0.5
h = img_hsv[:,:,0]
img_hsv[:, :, 0] = h*c_h
ax2[1].imshow(hsv2rgb(img_hsv))
ax2[1].axis('off')
ax2[1].set_title(f'Imagen HSV, H * {c_h}')

plt.tight_layout()

# Pruebas con S

img_hsv = rgb2hsv(img)

fig2, ax2 = plt.subplots(2, figsize=(4, 8))
ax2[0].imshow(img)
ax2[0].axis('off')
ax2[0].set_title('Imagen original')

c_s = 2
s = img_hsv[:,:,1]
img_hsv[:, :, 1] = s*c_s
ax2[1].imshow(hsv2rgb(img_hsv))
ax2[1].axis('off')
ax2[1].set_title(f'Imagen HSV, S * {c_s}')

plt.tight_layout()
plt.show()