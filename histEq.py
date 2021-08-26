import numpy as np


# Usar int() para tener intervalos finitos

# Histograma de imagen
def histograma(img, res=255):
    hist = np.zeros(res+1)
    n = res/max(img.flatten())
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[int(n*img[i,j])] += 1
    return hist

# Acumular un arreglo en otro de igual longitud
def acumulado(xs):
    acu = []
    suma = 0
    N = sum(xs)
    for x in xs:
        suma += x/N
        acu.append(suma)
    return acu

def ecualizar(img, res=255):
    acu = acumulado(histograma(img, res=res))
    img_eq = np.copy(img)
    max_val= max(img.flatten())
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            idx = int(res*img_eq[i, j]/max_val)
            img_eq[i, j] = acu[idx]*max_val
    return img_eq

