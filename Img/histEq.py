import numpy as np

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
    """
    Ecualización de histograma
    res: Subdivisiones del histograma
    """
    acu = acumulado(histograma(img, res=res))
    img_eq = np.zeros(img.shape)
    max_val = max(img.flatten())
    scale = res / max_val
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            idx = int(scale*img[i, j])
            img_eq[i, j] = acu[idx]*max_val
    return img_eq

from skimage import color
def trans_v(img_rgb, trans):
    img_hsv = color.rgb2hsv(img_rgb)
    v_trans = trans(np.copy(img_hsv[:,:,2]))
    img_hsv[:,:,2] = v_trans
    return color.hsv2rgb(img_hsv)

