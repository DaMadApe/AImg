"""
Crecimiento de región en una imagen
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, color, transform, filters
from skimage.morphology import binary_dilation
from matplotlib.colors import Normalize


def registrar_semillas(img):
    """
    Desplegar imagen para seleccionar semillas,
    devolver coordenadas
    """
    plt.figure()
    prev_ion = plt.isinteractive()
    plt.ion() # Activar interactividad
    plt.imshow(img, cmap='gray')
    seed = np.int32(plt.ginput(n=0, timeout=0))
    #plt.close()
    # Apagar interactividad si no estaba activada antes de función
    if not prev_ion:
        plt.ioff()
    return seed


def aislar_region(img, seed, tolerancia,
                  prom_fijo=True, visual=True):
    """
    Despliega la imagen para seleccionar las semillas,
    muestra el crecimiento de la región y devuelve
    la máscara de la región.

    seed: Coordenadas de semillas.
    tolerancia: Máxima diferencia entre promedio de
    región y nuevos pixeles.
    prom_fijo: Define si usar el promedio original de
    las semillas o el promedio instantáneo de región.
    visual: Define si se muestra el proceso de
    crecimiento. 
    """
    # Máscara de la región de interés
    region = np.zeros(img.shape, dtype=np.bool)
    region[seed[:, 1], seed[:, 0]] = True
    prom_region = np.mean(img[region])
    # Lienzo para dibujar la región de la imagen
    canvas = np.ones(img.shape)*255

    # Escalar tolerancia según escala de imagen
    tolerancia = tolerancia * img.max()

    if visual:
        # Preparar pyplot para mostrar el proceso
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        show_reg = ax[0].imshow(region)
        show_img = ax[1].imshow(canvas, cmap='gray',
                                norm=Normalize(0,img.max()))
        prev_ion = plt.isinteractive()
        plt.tight_layout()
        plt.ion() # Activar interactividad

    # Crecimiento de región
    while True:
        # Determinar la nueva frontera de la region
        borde = binary_dilation(region) & ~region

        # Obtener pixeles similares en la frontera
        if prom_fijo: # Usar solo promedio original
            dif = np.abs(img[borde] - prom_region)
        else: # Actualizar promedio de la región
            dif = np.abs(img[borde] - img[region].mean())
        similar = dif < tolerancia

        # Parar iteraciones si ya no hay nuevos pixeles
        if not similar.any():
            break
        # Agregar pixeles nuevos a la región
        region[borde] = similar

        if visual:
            # Actualizar imshow de máscara
            show_reg.set_data(region)
            # Actualizar imshow del cacho de imagen
            canvas[region] = img[region]
            show_img.set_data(canvas)
            
            plt.draw()
            plt.pause(0.0001)

    if visual:
        plt.waitforbuttonpress()
        # plt.close()
        # Apagar interactividad si no estaba activada antes de función
        if not prev_ion:
            plt.ioff()

    return region


"""
Pruebas
"""
if __name__ == '__main__':
    # path = "datos/paciente/23.JPG" # Tumor cerebral
    path = "datos/ultrasonido/ultra1.jpg" # Feto
    img = io.imread(path)
    if len(img.shape) > 2:
        img = color.rgb2gray(img)

    # Desplegar la pantalla para seleccionar las semillas
    seed = registrar_semillas(img)
    # Mostrar el crecimiento para tol=0.2
    aislar_region(img, seed, 0.1, visual=True) # 'visual' para mostrar proceso de crecimiento

    # Mostrar el resultado de las mismas semillas con múltiples valores de tolerancia
    regiones = []
    tolerancias = 0.025*np.arange(1, 9)
    for tol in tolerancias:
        regiones.append(aislar_region(img, seed, tol, visual=False, prom_fijo=False))

    # Probar distintas tolerancias
    plt.figure(figsize=(2*len(regiones), 8))
    for i, region in enumerate(regiones):
        plt.subplot(2, 4, i+1)
        plt.imshow(region)
        plt.axis('off')
        plt.title(f'tol = {tolerancias[i] :3f}')
    plt.tight_layout()
    plt.show()