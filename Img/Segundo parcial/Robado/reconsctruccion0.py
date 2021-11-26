from skimage import data, io , color,transform
import matplotlib.pyplot as plt
import numpy as np
from numpy import matlib

imagen=data.camera()/255.0
#agregar pixeles de ceros a la imagen 
filas1 = np.zeros((imagen.shape[0],130))
New=np.concatenate((filas1, imagen, filas1), axis=1)
columnas1 = np.zeros((130, New.shape[1]))
New = np.concatenate((columnas1, New, columnas1), axis=0)
plt.figure()
plt.imshow(New,cmap='gray')

#perfil=np.sum(imagen,axis=1)
#plt.figure()
#plt.plot(perfil)
#replica=np.matlib.repmat(perfil,512,1)
#plt.figure()
#plt.imshow(replica,cmap='gray')

suma=np.zeros((New.shape[0],New.shape[1]))
for i in range(0,180,1):        
    perfil=np.sum(transform.rotate(New,i),axis=0)
    replica=np.matlib.repmat(perfil,New.shape[0],1)
    suma+=transform.rotate(replica,i)
    


plt.figure()
plt.imshow(suma,cmap='gray')
plt.show()
