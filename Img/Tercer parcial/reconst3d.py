import numpy as np
from skimage import (io, util, color, transform,
                     filters, morphology)
from skimage.filters import rank

from rayosParalelos import *

"""
Procesar las imagenes de proyección y calcular transformación
de Radón para las rebanadas normales al eje de giro
"""
idx_inicio = 104 # Número de primera imagen (voltea a derecha)
idx_final = 150 # Número de última imagen (voltea a izquierda)
n_imgs = idx_final - idx_inicio +1

# Recorte de las imágenes
y0, y1 = 280, 800
x0, x1 = 10, 700
recorte = np.s_[y0:y1, x0:x1]

# Factor de reducción de las imágenes procesadas
escala = 10

rebanadas_radon = np.zeros(((y1-y0)//escala,
                            n_imgs,
                            (x1-x0)//escala), dtype=np.float16)

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
    # Reducir tamaño
    img_proc = transform.rescale(img_proc, 1/escala, anti_aliasing=True)

    # Sumar las rebanadas en este ciclo ahorra la necesidad
    # de tener todas las imágenes en memoria
    for f, fila in enumerate(img_proc):
        rebanadas_radon[f, idx] += fila

"""
Procesar la transformada de Radón de cada rebanada
"""
recons = np.zeros(((y1-y0)//escala,
                   (x1-x0)//escala,
                   (x1-x0)//escala), dtype=np.byte)
for i, rebanada in enumerate(rebanadas_radon):
    # Preprocesar en frecuencia
    proc = filtro_hamming(rebanada)
    # Reconstruir cada rebanada
    proc = inv_radon(proc)
    # Aplicar umbral para hacer binario
    proc = proc > 0.5*np.max(proc)
    recons[i] = proc

# Guardar la reconstrucción resultante
np.save('recons_craneo3d.npy', recons)