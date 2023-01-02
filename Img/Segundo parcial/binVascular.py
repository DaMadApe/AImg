import numpy as np
import matplotlib.pyplot as plt
# Todas las expresiones entre paréntesis las puedes separar en
# múltiples líneas, es útil si vas a importar muchos módulos!
from skimage import (io, data, exposure, util,
                     filters, morphology)
from skimage.morphology import disk
from skimage.filters import rank 


# Cargar imagen
#img = io.imread('datos/ojo/03_L.jpg')
img = data.retina()


# Pasos del procesamiento
proceso = []
# La idea es construir un arreglo como: [(img1, "descripción 1"), (img2, "descripción 2"), ...]
# Al final vas a ver por qué

# Paso 1
img_proc = img[:,:,1] # La ventaja de arreglos numpy es que puedes hacer magia negra como esta
                      # para sacar secciones de un arreglo, en este caso, sólo el canal verde
proceso.append((img_proc.copy(), 'Canal Verde'))
# Se usa img.copy() para que puedas seguir modificando la variable img_proc
# sin cambiar las imágenes que se están guardando en 'proceso'
# Esto se relaciona a algo llamado 'mutabilidad', y es una de las cosas
# que puede llegar a ser confusa en Python

# Paso 2
img_proc = util.img_as_ubyte(exposure.equalize_adapthist(img_proc))
proceso.append((img_proc.copy(), 'Ecualización adaptativa'))

# Paso 3
img_proc = util.img_as_ubyte(img_proc < rank.otsu(img_proc, disk(8)))
proceso.append((img_proc.copy(), 'Umbral local de Otsu r=8'))

# Paso 4
img_proc = rank.mean(img_proc, disk(1))
proceso.append((img_proc.copy(), 'Filtro medio r=1'))

# Paso 5
img_proc = util.img_as_ubyte(img_proc==255)
proceso.append((img_proc.copy(), 'Umbral de valor máximo'))

# Paso 6
img_proc = morphology.area_opening(img_proc, 800)
proceso.append((img_proc.copy(), 'Area opening A>800'))

# Prueba comentar cualquiera de los pasos, particularmente para
# acelerar la ejecución del código

# Mostrar imágenes
plt.imshow(img)
plt.title('Imagen original')

# Aquí se instancia la ventana para hacer las gráficas
plt.figure(figsize=(15, 10))

# Este ciclo es un ejemplo simpático de optimización:
# en lugar de escribir un 'imshow' para cada imagen,
# hacemos un bloque genérico para cada imagen en 'proceso'
for i, (img, titulo) in enumerate(proceso):
    # Esto es para trabajar sólo con una fracción de la ventana
    plt.subplot(2, 3, i+1)
    plt.imshow(img, cmap='gray')
    # Apagar las marcas de los ejes
    plt.axis('off')
    # Aquí es donde es útil el string de descripción de cada
    # imagen, para ponérselo de título
    plt.title(f'{i+1}. {titulo}')

# Reducir espacio entre imágenes
plt.tight_layout()

plt.show()