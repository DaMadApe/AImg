import numpy as np
from skimage import io, data, color, transform, util
import matplotlib.pyplot as plt


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


# Exposure region determination, Saad et al, 2020 (2)
def ERD(img, block_size):
    """
    Recibe imagen, devuelve arreglos booleanos de máscara
    para cada región de exposición
    Saad et al, 2020
    """
    #img_v = color.rgb2hsv(img)[:,:,2]
    # Promedio y desviación estándar de intensidad V
    v_avg = np.average(img) #(2,1)
    v_std = (np.sum((img-v_avg)**2)/img.size)**0.5 #(2,2)
    # Referencias de intensidad para categorización
    up_lim = v_avg + v_std #(2,3)
    low_lim = v_avg - v_std #(2,4)

    def entropy(img):
        hist, _ = np.histogram(img, bins=256,
                               range=(0, 255), density=True)
        e = 0
        for i in range(256):
            p_i = hist[i]
            if p_i > 0: # Evitar log(0) 
                e += -p_i*np.log2(p_i) #(2,5)
        return e

    def contrast(img):
        grey = img/255.0 # Revisar si es necesario dividir
        n = img.size
        c = np.sum(grey**2)/n - (np.sum(grey)/n)**2 #(2,9)
        return c

    # Entropía y contraste promedio de toda la imagen
    E_avg = 0
    C_avg = 0
    n_blocks = 0
    for loc in loc_gen(img, block_size):
        E_avg += entropy(img[loc])
        C_avg += contrast(img[loc])
        n_blocks += 1
    E_avg /= n_blocks
    C_avg /= n_blocks
    
    # Arreglos booleanos de máscara
    # La comparación devuelve un arreglo de 'False' del tamaño de img 
    UE = img < 0 # Región Under-exposed
    WE = img < 0 # Región Well-exposed
    OE = img < 0 # Región Over-exposed

    for loc in loc_gen(img, block_size):
        E = entropy(img[loc])
        C = contrast(img[loc])
        I = np.average(img[loc])

        if (E > E_avg and C > C_avg) or (I < up_lim and I > low_lim):
            WE[loc] = True
        elif I > up_lim:
            OE[loc] = True
        else:
            UE[loc] = True

    # Cambiar aleatoriamente un par de pixeles en cada conjunto
    # Solución floja para evitar los bugs de conjuntos vacíos :P
    random_idx = lambda x : (np.random.randint(0, x), np.random.randint(0, x))
    for _ in range(3):
        UE[random_idx(20)] = True
        WE[random_idx(20)] = True
        OE[random_idx(20)] = True

    return UE, WE, OE



def neimhe(img, block_size):
    """
    Nonlinaer Exposure Intensity based Modification Histogram Equalization, Saad et al, 2021 (1)
    """
    # Obtener regiones de exposición
    UE, WE, OE = ERD(img, block_size)

    # Obtener subregiones de región WE
    sub_we = np.zeros(img.shape, dtype=np.uint8) + np.average(img)
    sub_we[WE] = img[WE]

    LWE, MWE, UWE = ERD(sub_we, block_size)
    # Limitar las subregiones calculadas a su respectiva región
    LWE = np.logical_and(LWE, WE) # Región Lower Well Exposed 
    MWE = np.logical_and(MWE, WE) # Región Medium Well Exposed
    UWE = np.logical_and(UWE, WE) # Región Upper Well Exposed

    img_eq = np.copy(img)

    # Procesar cada región para obtener su respectiva escala correctiva
    # Cada región tiene un proceso similar:
    # 1. Calcular histograma y recortarlo.
    # 2. Calcular CDF ponderada según la región
    # 3. Calcular escala a partir de CDF según la región
    # 4. Con la siguiente función auxiliar se aplica cada escala a la imagen

    # Cambiar la escala de tonos en una región R de la imagen
    def apply_scale(img, R, scale):
        ecu = lambda i: scale[int(i)]
        # Vectorizar la traducción de tonos a la escala corregida
        ecu = np.vectorize(ecu)
        img[R] = ecu(img[R])

    # Procesar UE
    i_avg = np.average(img[UE])
    i_max = np.max(img[UE])
    i_min = np.min(img[UE])
    hist, _ = np.histogram(img[UE], bins=256,
                           range=(0, 255), density=True)
    hist = clip_hist(hist)

    def cdf_ue(k): #(1,1)
        r = 0.5 + 0.5*(i_avg-i_min)/(i_max-i_min) #(1,3)
        return (sum([hist[i] for i in range(k+1)]))**r

    cdf_ue_arr = np.array([cdf_ue(i) for i in range(256)])

    cr = 0.5 * i_max #(1,6)
    if cr < 51: #(1,4)
        ue_scale = (i_max + 51-cr - i_min)*cdf_ue_arr + i_min
    else:
        ue_scale = (i_max - i_min) * cdf_ue_arr + i_min

    ue_scale = np.rint(ue_scale)
    apply_scale(img_eq, UE, ue_scale)

    # Procesar OE
    i_avg = np.average(img[OE])
    i_max = np.max(img[OE])
    i_min = np.min(img[OE])
    hist, _ = np.histogram(img[OE], bins=256,
                           range=(0, 255), density=True)
    hist = clip_hist(hist)

    def cdf_oe(k): #(1,1)
        r = 0.5 + 0.5*(i_max-i_avg)/(i_max-i_min) #(1,3)
        return (sum([hist[i] for i in range(k, 256)]))**r

    cdf_oe_arr = np.array([cdf_oe(i) for i in range(256)])

    cr = i_min + 0.5 * (i_max - i_min) #(1,6)
    if cr > 204: #(1,5)
        oe_scale = i_max - (i_max - (i_min - (cr-204)))*cdf_oe_arr
    else:
        oe_scale = i_max - (i_max - i_min) * cdf_oe_arr

    oe_scale = np.rint(oe_scale)
    apply_scale(img_eq, OE, oe_scale)

    # Procesar LWE (usar cdf de UE)
    i_avg = np.average(img[LWE])
    i_max = np.max(img[LWE])
    i_min = np.min(img[LWE])
    hist, _ = np.histogram(img[LWE], bins=256,
                           range=(0, 255), density=True)
    hist = clip_hist(hist)

    def cdf_lwe(k): #(1,1)
        r = 0.5 + 0.5*(i_avg-i_min)/(i_max-i_min) #(1,3)
        return (sum([hist[i] for i in range(k+1)]))**r

    cdf_lwe_arr = np.array([cdf_lwe(i) for i in range(256)])

    lwe_scale = i_min + (ue_scale[-1] - i_min) * cdf_lwe_arr#(1,7)
    apply_scale(img_eq, LWE, lwe_scale)

    # Procesar MWE
    img_avg = np.average(img)
    hist, _ = np.histogram(img[MWE], bins=256,
                           range=(0, 255), density=True)
    hist = clip_hist(hist)

    # Esta sección es muy problemática, no sé si sea un error
    # del paper o un error de transcripción que no encuentro
    # Por eso vienen los ifs forzados con True
    def cdf_mwe(k): #(1,1)
        if True: #img_avg < 128:
            return sum([hist[i] for i in range(k+1)])
        else:
            return sum([hist[i] for i in range(k, 256)])
    cdf_mwe_arr = np.array([cdf_mwe(i) for i in range(256)])

    if True: #img_avg >= 128: #(1,7)
        mwe_scale = ue_scale[0] +(oe_scale[-1] - ue_scale[0])*cdf_mwe_arr
    else:
        mwe_scale = ue_scale[0] -(oe_scale[-1] - ue_scale[0])*cdf_mwe_arr

    apply_scale(img_eq, MWE, mwe_scale)

    # Procesar UWE
    i_avg = np.average(img[UWE])
    i_max = np.max(img[UWE])
    i_min = np.min(img[UWE])
    hist, _ = np.histogram(img[UWE], bins=256,
                           range=(0, 255), density=True)
    hist = clip_hist(hist)

    def cdf_uwe(k): #(1,1)
        r = 0.5 + 0.5*(i_max-i_avg)/(i_max-i_min) #(1,3)
        return (sum([hist[i] for i in range(k, 256)]))**r

    cdf_uwe_arr = np.array([cdf_uwe(i) for i in range(256)])
    
    uwe_scale = i_max - (i_max - oe_scale[0]) * cdf_uwe_arr #(1,7)
    apply_scale(img_eq, UWE, uwe_scale)

    return img_eq, (UE, WE, OE, LWE, MWE, UWE)



"""
Pruebas
"""

# Imagen ByN
img = data.astronaut()
img = util.img_as_ubyte(color.rgb2gray(img)) # Expresar en 0-255
img_eq, regs = neimhe(img, 5)
UE, WE, OE, LWE, MWE, UWE = regs

# Imagen RGB
# path = "datos/pulmones1.jpg"
# img = io.imread(path)
# #img = data.astronaut()
# #img = transform.rescale(img, (0.25, 0.25, 1)) #Para imágenes grandes
# img_hsv = color.rgb2hsv(img)
# img_v = img_hsv[:,:,2]

# img_v_eq, regs = neimhe(img_v*255, 5)
# UE, WE, OE, LWE, MWE, UWE = regs

# img_hsv[:,:,2] = img_v_eq/255
# img_eq = color.hsv2rgb(img_hsv)


# Graficar las regiones de exposición

# Visualizar las regiones resultantes
img_ue = np.zeros(img.shape[:2], dtype=np.uint8)
img_we = np.zeros(img.shape[:2], dtype=np.uint8)
img_oe = np.zeros(img.shape[:2], dtype=np.uint8)
img_ue[UE] = 255
img_we[WE] = 255
img_oe[OE] = 255
img_seg = np.dstack([img_ue, img_we, img_oe])

img_lwe = np.zeros(img.shape[:2], dtype=np.uint8)
img_mwe = np.zeros(img.shape[:2], dtype=np.uint8)
img_owe = np.zeros(img.shape[:2], dtype=np.uint8)
img_lwe[LWE] = 255
img_mwe[MWE] = 255
img_owe[UWE] = 255
img_seg_we = np.dstack([img_lwe, img_mwe, img_owe])

fig, ax = plt.subplots(2, 2)

ax[0, 0].imshow(img, cmap='gray')
ax[0, 0].axis('off')
ax[0, 0].set_title('Imagen original')
ax[0, 1].imshow(img_eq, cmap='gray')
ax[0, 1].axis('off')
ax[0, 1].set_title('Imagen modificada')

ax[1, 0].imshow(img_seg)
ax[1, 0].axis('off')
ax[1, 0].set_title('Regiones UE, WE, OE')
ax[1, 1].imshow(img_seg_we)
ax[1, 1].axis('off')
ax[1, 1].set_title('Regiones LWE, MWE, UWE')

plt.tight_layout()
plt.show()