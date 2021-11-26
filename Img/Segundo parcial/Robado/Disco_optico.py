from skimage import io, data,morphology,feature,util,exposure,filters,measure
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
#Imagen=data.retina()
Imagen=io.imread('datos/ojo/03_R.jpg')

verde=Imagen[:,:,1]
estructura=morphology.disk(10)

filtro=filters.rank.mean(verde,selem=estructura)


adaptable=exposure.equalize_adapthist(filtro)



fig,ejes=plt.subplots(2,2)
fig.suptitle('Procesamiento del ojo')
ejes[0,0].imshow(Imagen)
ejes[0,1].imshow(verde,cmap='gray')
ejes[1,0].imshow(filtro,cmap='gray')
ejes[1,1].imshow(adaptable,cmap='gray')

lado=2#int(input('mensionar si el ojo es el izquierdo o el derecho:(1=izquierdo)(2=derecho) '))
fil,col=adaptable.shape
if(lado==1):   
    recorte=adaptable[int(fil/5):int(fil*4/5),0:int(col/2)]
elif(lado==2):
    recorte=adaptable[int(fil/5):int(fil*4/5),int(col/2):-1]
plt.figure()
plt.imshow(recorte,cmap='gray') 
disco=morphology.disk(10)
open_=morphology.opening(recorte,disco)
close=morphology.closing(open_,disco)
maximo=np.max(close)
if(maximo<1):
    a=1-maximo
    close=close+a
binario=(close>.98).astype(int)
plt.figure()
plt.imshow(binario,cmap='gray')
label=measure.label(binario)
centroide=measure.regionprops(label)
for props in centroide:
    y,x=props.centroid

if(lado==1):
    y0=y+int(fil/5)
    x0=x
else:
    y0=y+int(fil/5)
    x0=x+int(col/2)

for ii in range(Imagen.shape[0]):
    for jj in range (Imagen.shape[1]):
        radio = ( (ii - y0)**2 + (jj - x0)**2 )        
        if (radio < 20000):
            Imagen[ii, jj, :] = 0
plt.figure()
plt.imshow(Imagen)

plt.show()





