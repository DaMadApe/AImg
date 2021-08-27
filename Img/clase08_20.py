import skimage
from skimage import color
import numpy as np
import matplotlib.pyplot as plt

img = skimage.data.retina()
plt.figure()
plt.imshow(img)

gris1 = color.rgb2gray(img)
#plt.figure()
#plt.imshow(gris1, cmap='gray')

rojoNuevo = 0.8*img[:,:,0] + 0.1*img[:,:,1] + 0.1*img[:,:,2]
verdeNuevo = 0.45*img[:,:,0] + 0.1*img[:,:,1] + 0.45*img[:,:,2]
azulNuevo = 0.8*img[:,:,0] + 0.1*img[:,:,1] + 0.1*img[:,:,2]
plt.figure()
#plt.imshow(gris2, cmap='gray')
plt.show()