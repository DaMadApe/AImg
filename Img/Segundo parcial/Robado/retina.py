from skimage import io, data,morphology,feature,util,exposure,filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
Imagen=data.retina()
#plt.figure()
#plt.imshow(Imagen)
rojo=Imagen[:,:,0]
verde=Imagen[:,:,1]
azul=Imagen[:,:,2]
#plt.figure()
#plt.imshow(rojo,cmap='gray')
plt.figure()
plt.imshow(verde,cmap='gray') # 1
#plt.figure()
#plt.imshow(azul,cmap='gray')
inversa=util.invert(verde)
plt.figure()
plt.imshow(inversa,cmap='gray') # 2

adaptable=exposure.equalize_adapthist(inversa)
plt.figure()
plt.imshow(adaptable,cmap='gray') # 3

estructura=morphology.disk(10)
apertura=morphology.opening(adaptable,estructura)
plt.figure()
plt.imshow(apertura,cmap='gray') # 4

resta=adaptable-apertura
plt.figure()
plt.imshow(resta,cmap='gray') # 5

filtro=filters.median(resta,morphology.disk(3))
salida=exposure.adjust_gamma(filtro,2)
plt.figure()
plt.imshow(filtro,cmap='gray') # 6

plt.figure()
plt.imshow(salida,cmap='gray') # 7
umbral=filters.threshold_otsu(salida)
binaria=(salida>umbral).astype(int)
plt.figure()
plt.imshow(binaria,cmap='gray') # 8

plt.show()