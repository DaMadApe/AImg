import numpy as np
from skimage import data, color, io
import skfuzzy as fuzzy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Funciones auxiliares del programa

def histograma(img):
    # Histograma normalizado de imagen de un canal
    hist = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            idx = int(img[i,j])
            hist[idx] += 1
    return hist/sum(hist)


def acumulado(xs):
    # Acumular un arreglo en otro de igual longitud
    return [sum(xs[0:i+1]) for i in range(len(xs))]


def ecu_hist(img):
    """
    Ecualización de histograma de un sólo canal
    """
    acu = acumulado(histograma(img))
    img_eq = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            idx = int(img[i, j])
            img_eq[i, j] = np.uint8(acu[idx]*255)
    return img_eq


def ecu_fuzzy(img, singl):
    """
    Ecualización difusa de histograma
    singl: vector de singletons de conjuntos
    """
    # Conjuntos difusos
    tone_range = np.arange(256)
    oscuros = fuzzy.zmf(tone_range, 15, 130)
    grises = fuzzy.gbellmf(tone_range, 50, 3, 130)
    claros = fuzzy.smf(tone_range, 130, 230)
    fuzzy_sets = np.stack([oscuros, grises, claros])

    ecufuzz = np.zeros(256)
    for i in range(256):
        ecufuzz[i] = np.dot(fuzzy_sets[:, i], singl)/sum(fuzzy_sets[:,i])

    img_eq = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            idx = int(img[i, j])
            img_eq[i, j] = np.uint8(ecufuzz[idx])
    return img_eq


path = 'datos/pulmones1.jpeg'
#img = data.camera()/3 + 80
img = io.imread(path)
img = color.rgb2gray(img)*255

# Imagen con ecualización normal
img_he = ecu_hist(img)

# Singletons de conjuntos difusos
s1 = 30
s2 = 110
s3 = 200
singl = [s1, s2, s3]
# Imagen con ecualización difusa
img_fhe = ecu_fuzzy(img, singl)



# Comparar métodos de ecualización con la misma imagen
fig1, ax1 = plt.subplots(2, 3, figsize=(12,8))
ax1[0, 0].imshow(img, cmap='gray', norm=Normalize(0, 255))
ax1[0, 0].axis('off')
ax1[0, 0].set_title('Imagen original')
ax1[0, 1].imshow(img_he, cmap='gray', norm=Normalize(0, 255))
ax1[0, 1].axis('off')
ax1[0, 1].set_title('Ecualización de hist normal')
ax1[0, 2].imshow(img_fhe, cmap='gray', norm=Normalize(0, 255))
ax1[0, 2].axis('off')
ax1[0, 2].set_title(f'Ecualización fuzzy, S={singl}')

ax1[1, 0].plot(histograma(img))
ax1[1, 1].plot(histograma(img_he))
ax1[1, 2].plot(histograma(img_fhe))


# Comparar el efecto de diferentes singletons
singls_prueba = [[0, 128, 255],
                 [15, 120, 240],
                 [10, 100, 180]]

fig2, ax2 = plt.subplots(1, len(singls_prueba)+1)

ax2[0].imshow(img, cmap='gray', norm=Normalize(0, 255))
ax2[0].axis('off')
ax2[0].set_title('Imagen original')

for i, singl in enumerate(singls_prueba):
    ax2[i+1].imshow(ecu_fuzzy(img, singl), cmap='gray', norm=Normalize(0, 255))
    ax2[i+1].axis('off')
    ax2[i+1].set_title(f'S={singl}')

plt.tight_layout()
plt.show()