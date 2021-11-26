from skimage.morphology import skeletonize
from mpl_toolkits.mplot3d import Axes3D

import funciones as func
import numpy as np
import matplotlib.pyplot as plt

#Ima = plt.imread('segmentos/completa.bmp')[:,:,0]/255
Ima1   = plt.imread('02_h/3D1.bmp')[:,:,1]
[f,c] = Ima1.shape
Ima   = Ima1 > 0.5

#Genera bordes para evitar errores
Ima[0:3,:]   = 0
Ima[f-3:f,:] = 0

Orig = Ima


Ima = skeletonize(Ima)

curvasx = []
curvasy = []
min_len = 21

#-------------------------------------------------------------
#     Obtiene los vectores con las venas ordenados
#-------------------------------------------------------------
test = 0

while np.sum(Ima) > 0:
    
    i_x,i_y = func.extremo(Ima) 
    if i_x > 0 and i_y > 0:
        Ima,xcoord,ycoord = func.ordena(Ima,i_x,i_y,min_len)
    
        if len(xcoord) > min_len:
            curvasx.append(xcoord)
            curvasy.append(ycoord)
        
        print(test)
        test += 1
        
    else:
        break
            

#-------------------------------------------------------------
# Muestra la imagen de cada region de los vectores ordenados
#-------------------------------------------------------------


ordenados = np.zeros((f,c,3))
for j in range(len(curvasx)):
    xcoord = curvasx[j]
    ycoord = curvasy[j]
    color = np.array([np.random.rand(),np.random.rand(),np.random.rand()])*0.7 + 0.3
    for i in range(len(xcoord)):
        ordenados[int(xcoord[i]),int(ycoord[i]),:] = color


plt.figure(1)
plt.imshow(Orig,cmap='gray')
plt.title('Imagen original')

plt.figure(2)
plt.imshow(ordenados)
plt.title('Venas segmentadas')

plt.figure(3)
plt.imshow(Orig,cmap = 'gray')
plt.title('Polinomios evaluados')



#-------------------------------------------------------------
#     Obtiene los coeficientes de los polinomios
#-------------------------------------------------------------

# Se crea una lista con los coeficientes

coeficientes = []
num_venas = len(curvasx)
for i in range(num_venas): 
    
    # Obtiene los coeficientes
    [ppx,ppy] = func.get_poly(curvasy[i],curvasx[i])
    coeficientes.append([ppx,ppy])
    
    

#-------------------------------------------------------------
#    Evalua los polinomios con un numero dijo de valores
#-------------------------------------------------------------

# en una lista  (venas)
n = 10
venas = []
    
for j in range(num_venas):
    
    [coef_x,coef_y] = coeficientes[j]
    plt.figure(3)
    #pp.c son los coeficientes, se usan para formar de nuevo la curva:
    for i in range(len(coef_x.x)-1):
        s = np.linspace(coef_x.x[i], coef_x.x[i+1], n)
        xx = np.polyval(coef_x.c[:,i], s - coef_x.x[i])
        yy = np.polyval(coef_y.c[:,i], s - coef_y.x[i])
        venas.append([xx,yy])
        plt.plot(xx,yy,'r')
    
    plt.xlim([0,c])
    plt.ylim([0,f])
    

#-------------------------------------------------------------
#                 Realiza la proyección en 3D
#-------------------------------------------------------------
  
    
lims = np.sum(1-Orig,0) > 5

# Radio de la esfera
r = np.fix(np.sum(lims)/2) + 200

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')

for i in range(len(venas)):
    
    [yp,xp] = venas[i]
    xp = xp - f/2 
    yp = yp - c/2 + 150
    
    verifica = xp**2 + yp**2 <= r**2;
    pos = np.where(verifica == 1)
    
    x = xp[pos] 
    y = yp[pos] 
    z = np.sqrt( r**2 - (x**2) - (y**2)  )
    
    
    ax.plot(-z,y,x,color='r')
    ax.auto_scale_xyz([-500, 500], [-700, 700], [-500, 500])
#    
    



