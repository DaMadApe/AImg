import numpy as np
import matplotlib.pyplot as plt
from skimage import (io, data, util, color,
                     filters, morphology)
from skimage.filters import rank
from scipy.spatial.transform import Rotation as R

"""
Obtener contornos de cada imagen
Muestrear ~100 puntos del contorno
Guardar con el ángulo codificado
"""
rot = np.pi/46
angles = [rot*i for i in range(99, 202)]
contornos = []

for indx in range(99, 202):
    # Cargar imagen
    img = io.imread(f'datos/craneo_3d/IMG_0745 ({indx}).jpg')

    # Monocromatización y recorte manual
    img_proc = util.img_as_ubyte(color.rgb2gray(img))
    img_proc = img_proc[250:850]
    # Filtro mediano
    img_proc = rank.median(img_proc, morphology.disk(8))
    # Umbral de Otsu
    img_proc = util.img_as_ubyte(img_proc < filters.threshold_otsu(img_proc))

    borde = morphology.binary_dilation(img_proc) & ~img_proc

    puntos = np.argwhere(borde) # ~1500 puntos
    puntos = puntos[::15] # Tomar ~100 puntos por contorno

    contornos.append(puntos)

"""
Rotar conjuntos de puntos
"""

nube_puntos = np.array([[0,0,0]])
for i, puntos in enumerate(contornos):
    # Agregar valor z a cada coordenada
    n = puntos.shape[0]
    puntos = np.concatenate((puntos, np.zeros((n,1))), axis=1)

    # Generar matriz de rotación en torno a x, y desplazamientos
    rot = R.from_rotvec((angles[i], 0, 0)).as_matrix()
    tr1 = np.array([0, -320, 0])
    tr2 = np.array([0, -320, 0])
    # Rotar y desplazar cada punto de los contornos
    for punto in range(n):
        puntos[punto] = np.dot(rot, puntos[punto]+tr1) # +tr2

    nube_puntos = np.concatenate((nube_puntos, puntos), axis=0)


ax = plt.axes(projection='3d')
ax.scatter(*nube_puntos.T)
plt.show()