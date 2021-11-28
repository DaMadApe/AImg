import numpy as np
import matplotlib.pyplot as plt
from skimage import (io, util, color, transform,
                     filters, morphology)
from skimage.filters import rank
from scipy.spatial.transform import Rotation as R

from rayosParalelos import *

"""
Procesar las imagenes de proyección y calcular transformación
de Radón para las rebanadas normales al eje de giro
"""
idx_inicio = 104 # Número de primera imagen (voltea a derecha)
idx_final = 150 # Número de última imagen (voltea a izquierda)
n_imgs = idx_final - idx_inicio +1
# Recorte de las imágenes
y0, y1 = 270, 830
x0, x1 = 10, 700
recorte = np.s_[y0:y1, x0:x1]
h_rebanada = 30
rebanadas_radon = np.zeros(((y1-y0)//h_rebanada, n_imgs, x1-x0), dtype=np.float16)

for idx in range(0, n_imgs):
    # Cargar imagen
    img = io.imread(f'datos/craneo_3d/IMG_0745 ({idx+idx_inicio}).jpg')
    # Monocromatización y recorte manual
    img_proc = util.img_as_ubyte(color.rgb2gray(img))
    img_proc = img_proc[recorte]
    # Filtro mediano
    img_proc = rank.median(img_proc, morphology.disk(8))
    # Umbral de Otsu
    img_proc = util.img_as_ubyte(img_proc < filters.threshold_otsu(img_proc))

    # Sumar las rebanadas en este ciclo ahorra la necesidad
    # de tener todas las imágenes en memoria
    for i in range(rebanadas_radon.shape[0]):
        filas = np.s_[h_rebanada*i:h_rebanada*(i+1)]
        rebanada = np.average(img_proc[filas], axis=0)
        rebanadas_radon[i, idx] += rebanada

"""
Procesar el sinograma de cada rebanada
"""
recons = np.zeros((y1-y0, x1-x0, x1-x0), dtype=np.byte)
for i, rebanada in enumerate(rebanadas_radon):
    proc = filtro_hamming(rebanada)
    proc = inv_radon(proc) > 0.5
    filas = np.s_[h_rebanada*i:h_rebanada*(i+1)]
    recons[filas] = proc * np.ones((h_rebanada,1,1), dtype=np.byte)


np.save('recons_craneo3d.npy', recons)