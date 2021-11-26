from skimage import io, data,morphology,feature
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np

#--------------------------------------------------------------
#Image = io.imread('lena.png')
Image = io.imread('Brain.png')
Gris = rgb2gray(Image)*255
binario=(Gris<94)
plt.figure()
plt.imshow(Gris,cmap='gray')
plt.figure()
plt.imshow(binario,cmap='gray')
estructura=morphology.disk(3)
#open_=morphology.opening(binario,estructura)
open_=morphology.closing(binario,estructura)

plt.figure()
plt.imshow(open_,cmap='gray')
binario3=Gris*open_
plt.figure()
plt.imshow(binario3,cmap='gray')
bordes=feature.canny(binario3,sigma=3)
plt.figure()
plt.imshow(bordes,cmap='gray')
salida=(bordes*255)+Gris

plt.figure()
plt.imshow(salida,cmap='gray')


