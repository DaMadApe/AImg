import numpy as np
from numpy.core.shape_base import block
import imglib as lib
from skimage import io, data, color
import matplotlib.pyplot as plt


# Exposure region determination, Saad et al, 2020 (2)
def ERD(img, block_size):
    """
    Recibe imagen, devuelve
    """
    #img_v = color.rgb2hsv(img)[:,:,2]
    # Promedio y desviación estándar de intensidad V
    v_avg = np.average(img) #(2,1)
    v_std = (np.sum((img-v_avg)**2)/img.size)**0.5 #(2,2)
    # Referencias de intensidad para categorización
    up_lim = v_avg + v_std #(2,3)
    low_lim = v_avg - v_std #(2,4)

    def entropy(img):
        hist = lib.histograma(img)
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
    for loc in lib.loc_gen(img, block_size):
        E_avg += entropy(img[loc])
        C_avg += contrast(img[loc])
        n_blocks += 1
    E_avg /= n_blocks
    C_avg /= n_blocks
    
    # Arreglos booleanos de máscara
    # Inicializar del tamaño de la imagen y con falsos
    UE = img < 0 # Under-exposed
    WE = img < 0 # Well-exposed
    OE = img < 0 # Over-exposed

    for loc in lib.loc_gen(img, block_size):
        E = entropy(img[loc])
        C = contrast(img[loc])
        I = np.average(img[loc])

        if (E > E_avg and C > C_avg) or (I < up_lim and I > low_lim):
            WE[loc] = True
        elif I > up_lim:
            OE[loc] = True
        else:
            UE[loc] = True

    return UE, WE, OE


def neimhe(img, block_size):

    # Obtener regiones de exposición
    UE, WE, OE = ERD(img, block_size)

    # Obtener subregiones de región WE
    sub_we = np.zeros(img.shape, dtype=np.uint8) + np.average(img)
    sub_we[WE] = img[WE]
    LWE, MWE, UWE = ERD(sub_we, block_size)
    # Limitar las subregiones calculadas a su respectiva región
    LWE = np.logical_and(LWE, WE)
    MWE = np.logical_and(MWE, WE)
    UWE = np.logical_and(UWE, WE)

    # Procesar cada región para obtener su respectiva escala correctiva

    # Función auxiliar para aplicar una escala a su respectiva región
    def apply_scale(img, R, scale):
        ecu = lambda idx: scale[idx]
        ecu = np.vectorize(ecu)
        img[R] = ecu(img[R])

    img_eq = np.copy(img)

    # Procesar UE
    i_avg = np.average(img[UE])
    i_max = np.max(img[UE])
    i_min = np.min(img[UE])
    hist, _ = np.histogram(img[UE], bins=256,
                           range=(0, 255), density=True)
    hist = lib.clip_hist(hist)

    def cdf_ue(k): #(1,1)
        r = 0.5 + 0.5*(i_avg-i_min)/(i_max-i_min) #(1,3)
        return (sum([hist[i] for i in range(k+1)]))**r

    cdf_ue_arr = np.array([cdf_ue(i) for i in range(256)])
    print(cdf_ue_arr[-5:-1])

    cr = 0.5 * i_max #(1,6)
    if cr < 51:
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
    hist = lib.clip_hist(hist)

    def cdf_oe(k): #(1,1)
        r = 0.5 + 0.5*(i_max-i_avg)/(i_max-i_min) #(1,3)
        return (sum([hist[i] for i in range(k, 256)]))**r

    cdf_oe_arr = np.array([cdf_oe(i) for i in range(256)])
    print(cdf_oe_arr[1:5])

    cr = i_min + 0.5 * (i_max - i_min) #(1,6)
    if cr > 204:
        oe_scale = i_max - (i_max - (i_min - (cr-204)))*cdf_oe_arr
    else:
        oe_scale = i_max - (i_max - i_min) * cdf_oe_arr

    oe_scale = np.rint(oe_scale)
    apply_scale(img_eq, OE, oe_scale)

    # Procesar LWE (usar cdf de UE)
    i_min = np.min(img[LWE])

    lwe_scale = i_min + (ue_scale[-1] - i_min) * cdf_ue_arr
    apply_scale(img_eq, LWE, lwe_scale)

    # Procesar MWE
    img_avg = np.average(img)
    hist, _ = np.histogram(img[MWE], bins=256,
                           range=(0, 255), density=True)
    hist = lib.clip_hist(hist)

    def cdf_mwe(k): #(1,1)
        if img_avg < 128:
            return sum([hist[i] for i in range(k+1)])
        else:
            return sum([hist[i] for i in range(k, 256)])
    cdf_mwe_arr = np.array([cdf_mwe(i) for i in range(256)])
    print(cdf_mwe_arr[-5:-1])

    if img_avg >= 128:
        mwe_scale = ue_scale[0] +(oe_scale[0] - ue_scale[-1])*cdf_mwe_arr
    else:
        mwe_scale = oe_scale[0] -(oe_scale[0] - ue_scale[-1])*cdf_mwe_arr

    apply_scale(img_eq, MWE, mwe_scale)

    # Procesar UWE (usar cdf de OE)
    i_max = np.max(img[UWE])
    
    uwe_scale = i_max - (i_max - oe_scale[0]) * cdf_oe_arr
    apply_scale(img_eq, UWE, uwe_scale)

    return img_eq, (UE, WE, OE, LWE, MWE, UWE)



img = data.camera()

img_eq, regs = neimhe(img, 10)
UE, WE, OE, LWE, MWE, UWE = regs

# Graficar las regiones de exposición

# Visualizar las regiones resultantes
img_ue = np.zeros(img.shape, dtype=np.uint8)
img_we = np.zeros(img.shape, dtype=np.uint8)
img_oe = np.zeros(img.shape, dtype=np.uint8)
img_ue[UE] = 255
img_we[WE] = 255
img_oe[OE] = 255
img_seg = np.dstack([img_ue, img_we, img_oe])

img_lwe = np.zeros(img.shape, dtype=np.uint8)
img_mwe = np.zeros(img.shape, dtype=np.uint8)
img_owe = np.zeros(img.shape, dtype=np.uint8)
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