import numpy as np
import skfuzzy as fuzzy
from skimage import color


# def histograma(img_in, res=255):
#     # Histograma normalizado de imagen de un canal
#     scale = res/max(img_in.flatten())
#     img = np.rint(scale*img_in)
#     hist = np.zeros(res+1)
    
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             idx = int(img[i,j])
#             hist[idx] += 1
#     return hist/sum(hist)

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


def ecu_hist(img, res=255):
    """
    Ecualización de histograma de un sólo canal
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
    singl: vector de singletons de conjuntos
    fuzzy_sets: Funciones de membresía por conjunto
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


def loc_gen(img, step: int):
    """
    Generador de secciones cuadradas para imagen de 1 canal
    Genera rebanadas np._s para usarlos como img[loc]
    shape: Tamaño total a subdividir
    step: Lado de sección cuadrada
    """
    fil = int(img.shape[0]/step)
    col = int(img.shape[1]/step)

    for f in range(fil-1):
        for c in range(col-1):
            yield np.s_[step*f: step*(f+1), step*c: step*(c+1)]

    #Bloques al margen cubren el sobrante del paso
    ext_f = int(img.shape[0])%step
    ext_c = int(img.shape[1])%step
    #Recorrido de bloques al margen derecho
    for f in range(fil-1):
        yield np.s_[step*f: step*(f+1), -(step + ext_c): ]

    #Recorrido de bloques al margen inferior
    for c in range(col-1):
        yield np.s_[-(step + ext_f): , step*c: step*(c+1)]

    #Esquina inferior derecha, caso mínimo
    yield np.s_[-(step + ext_f): , -(step + ext_c): ]


# Limitar histograma al número promedio de pix por bin
def clip_hist(hist):
    avg = np.average(hist)
    clip = np.copy(hist)
    extra = 0
    for i, bin in enumerate(hist):
        if bin > avg:
            extra += bin - avg
            clip[i] = avg
    clip += extra/len(clip)
    return clip