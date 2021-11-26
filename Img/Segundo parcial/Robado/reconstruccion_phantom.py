import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
from skimage import data,transform
plt.close('all')

def filtro_proyeccion(transformada):
    a=1#factor de amplificacion
    longitud=len(transformada[0])
    tetha=len(transformada)
    vector=np.arange( -np.pi, np.pi-(2*np.pi)/(longitud+1), (2*np.pi)/(longitud+1) )
    rn1 = np.abs( 2/a*np.sin(a*vector/2) )
    rn2 = np.sin(a*vector/2)
    rd = (a*vector)/2
    r = rn1 * (rn2/rd)**2#filtro
    f = np.fft.fftshift(r)#transformada de fourier del filtro
    salida=np.zeros((tetha,longitud))
    for i in range(tetha):
        fourier=np.fft.fft(transformada[i])
        convolucion=fourier*f
        salida[i,:]=np.real(np.fft.ifft(convolucion))
    return salida

imagen=data.shepp_logan_phantom()
suma=np.zeros((imagen.shape[0],imagen.shape[1]))
tradon=[]
for tetha in range(0,180,1):
    tradon.append(np.sum(transform.rotate(imagen,tetha),axis=0))
    replica=np.matlib.repmat(tradon[-1],imagen.shape[0],1)
    suma=transform.rotate(replica,tetha)+suma
plt.figure()
plt.subplot(1,4,1)
plt.imshow(suma,cmap='gray')

plt.subplot(1,4,2)
plt.imshow(tradon,cmap='gray')

filtrado=filtro_proyeccion(tradon)

plt.subplot(1,4,3)
plt.imshow(filtrado,cmap='gray')

suma1=np.zeros((imagen.shape[0],imagen.shape[1]))

for tetha in range(0,180,1):
    replica=np.matlib.repmat(filtrado[tetha],imagen.shape[0],1)
    suma1=transform.rotate(replica,tetha)+suma1

plt.subplot(1,4,4)
plt.imshow(suma1,cmap='gray')

plt.show()