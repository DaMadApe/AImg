import numpy as np
import skfuzzy as fuzzy
from skimage import color


def histograma(img, res=255):
    # Histograma normalizado de imagen
    hist = np.zeros(res+1)
    scale = res/max(img.flatten())
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            idx = int(scale*img[i,j])
            hist[idx] += 1
    return hist/sum(hist)


def acumulado(xs):
    # Acumular un arreglo en otro de igual longitud
    return [sum(xs[0:i+1]) for i in range(len(xs))]


def ecu_hist(img, res=255):
    """
    Ecualización de histograma
    res: Subdivisiones del histograma
    """
    acu = acumulado(histograma(img, res=res))
    max_val = max(img.flatten())
    scale = res / max_val
    img_eq = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            idx = int(scale*img[i, j])
            img_eq[i, j] = np.uint8(acu[idx]*max_val)
    return img_eq


def ecu_fuzzy(img, singl, fuzzy_sets, res=255):
    """
    Ecualización difusa de histograma
    res: Subdivisiones del histograma
    """
    ecufuzz = np.zeros(res+1)
    for i in range(res):
        ecufuzz[i] = np.dot(fuzzy_sets[:, i], singl)/sum(fuzzy_sets[:,i])

    img_eq = np.zeros(img.shape)
    max_val = max(img.flatten())
    scale = res / max_val
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            idx = int(scale*img[i, j]) #scale*
            img_eq[i, j] = np.uint8(ecufuzz[idx])
    return img_eq


def trans_v(img_rgb, trans):
    """
    Convertir RGB->HSV, aplicar trans a canal V,
    devolver RGB 
    """
    img_hsv = color.rgb2hsv(img_rgb)
    v_trans = trans(np.copy(img_hsv[:,:,2]))
    img_hsv[:,:,2] = v_trans
    return color.hsv2rgb(img_hsv)