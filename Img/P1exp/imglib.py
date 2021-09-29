import numpy as np

"""
Archivo de funciones redactadas para manipular
imágenes, incluidos los métodos de mejora
utilizados. 
"""

def img_as_uint8(img_in):
    # Convierte imagen a np.array con rango 0-255
    scale = 255/np.max(img_in)
    img_out = np.array(img_in)
    img_out = np.rint(scale*img_out)
    img_out = np.uint8(img_out)
    return img_out


def histograma(img_in, res=255):
    # Histograma normalizado de imagen de un canal
    scale = res/np.max(img_in)
    img = np.rint(scale*img_in)
    hist = np.zeros(res+1)
    
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            idx = np.uint8(img[i][j])
            hist[idx] += 1
    return hist/sum(hist)


def acumulado(xs):
    # Acumular un arreglo en otro de igual longitud
    return [sum(xs[0:i+1]) for i in range(len(xs))]


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
    

def hist_stretch(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return (img-img_min) * 255/(img_max-img_min)


def ecu_hist(img, res=255):
    """
    Ecualización de histograma de un sólo canal
    res: Subdivisiones del histograma
    """
    hist = histograma(img, res=res)
    acu = acumulado(hist)
    max_val = np.max(img)
    scale = res / max_val
    img_eq = np.zeros(np.shape(img))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            idx = int(scale*img[i][j])
            img_eq[i, j] = np.uint8(acu[idx]*max_val)
    return img_eq




def loc_gen(img, step: int):
    """
    Generador de secciones cuadradas para indexar imagen
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