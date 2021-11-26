from skimage import io, data
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
#--------------------------------------------------------------
#Image = io.imread('lena.png')
Image = io.imread('Brain.png')
Gris = rgb2gray(Image)*255
#Gris = data.camera()
plt.figure(1)
plt.imshow(Image)
plt.figure(2)
plt.imshow(Gris, cmap = 'gray')
plt.show()
#-------------- Calcular histogrma y el vector de probabilidades -------------
histo = np.zeros(256)
fil, col = Gris.shape
for i in range (fil):
    for j in range(col):
        pos = np.int(Gris[i,j])
        histo[pos] += 1
pro = histo / (fil*col)
#----------------------------------------------------------------
#----------- Momentos acumulativos ----------------------------
P = np.zeros(256)
S = np.zeros(256)
P[0] = pro[0]
S[0] = 0 * pro[0]

for v in range(len(pro)-1):
    P[v+1] = P[v] + pro[v+1]
    S[v+1] = S[v] + (v+1) * pro[v+1]
    #Calculo del vector
#--------------------------------------------------------------------  
#-------- Calculo de la Matriz de momentos acumulativos -----------------
PP = np.zeros( (256,256) )
SS = np.zeros( (256,256) )
HH = np.zeros( (256,256) )
Resta1 = np.zeros(len(pro)+2)
Resta2 = np.zeros(len(pro)+2)
Resta1[1:-1] = P
Resta2[1:-1] = S

for u in range(256):
    for v in range(256):
        PP[u,v] = P[v] - Resta1[u] + 0.00000001
        SS[u,v] = S[v] - Resta2[u] + 0.00000001
        HH[u,v] = (SS[u,v] * SS[u,v]) / PP[u,v]
#---------- Calculo de los umbrales ----------------------------------
u = 0
M = 3#numero de clases
L = 255#maximos niveles de grises
for t1 in range (0, L - (M-1), 1):
    if (M == 2):
        R1 = HH[0,t1] + HH[t1 + 1,L]
        r = R1
        if (u < r):
            u = r
            umbral = t1-1
    if (M==3):
        for t2 in range (t1+1, L-(M-2), 1):
            R1 = HH[1,t1] + HH[t1+1, t2] + HH[t2+1, L]
            r = R1
            if (u < r):
                u = r
                umbral = np.array([t1, t2])-1
print(umbral)
