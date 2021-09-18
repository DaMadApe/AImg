import numpy as np
import cv2
from skimage import io, data
from skimage.filters import threshold_mean
import matplotlib.pyplot as plt


def clip_letrs(txt_img):
    """
    Separar una imagen de texto en un arreglo
    de las letras encuadradas.
    """
    # Hacer binaria la imagen, sólo 0 o 255
    _, thresh_img = cv2.threshold(txt_img, 128, 255, cv2.THRESH_BINARY)
    # Invertir imagen, cv2 funciona bien con letras blancas en fondo negro
    thresh_img = 255 - thresh_img
    # Encontrar los contornos de la imagen
    contornos, _ = cv2.findContours(thresh_img,
                                    mode=cv2.RETR_EXTERNAL,
                                    method=cv2.CHAIN_APPROX_SIMPLE)
    letras = []
    for cnt in contornos:
        # Encuadrar el contorno
        x,y,w,h = cv2.boundingRect(cnt)
        letra = np.copy(txt_img[y:y+h, x:x+w]) # Considerar w,h constantes para imágenes de mismo tamaño
        letras.append(letra)
    return letras


def igualar_tamaño(imgs):
    """
    Agregar espacio en blanco para hacer igual el
    tamaño de todas las imágenes en un conjunto
    """
    max_size = max([max(img.shape) for img in imgs])
    imgs_copy = np.zeros((len(imgs), max_size, max_size))
    for i, img in enumerate(imgs):
        imgs_copy[i, img.shape] = np.copy(img)
    return imgs_copy

# Importar la imagen en escala de grises
img = cv2.imread('datos/arial_font.png', 0)
letras = igualar_tamaño(clip_letrs(img))

plt.figure()
plt.imshow(img, cmap='gray')
plt.figure()
for i in range(9):
    plt.subplot(331+i)
    plt.imshow(letras[-i], cmap='gray')
    plt.title(f"cd -{i}")

plt.show()

# cv2.imshow("fuente", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()