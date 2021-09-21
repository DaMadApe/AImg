import numpy as np
import cv2
from skimage import io, data, util
from skimage.filters import threshold_mean
import matplotlib.pyplot as plt


def leer_imagen(txt_img, modelo):
    """
    Producir el texto contenido en una imagen.
    Devuelve una cadena con el contenido y un arreglo
    con las dimensiones de cada letra en la imagen.

    modelo: Función usada para identificar cada imagen
    de letra con un caracter.
    """
    # Hacer binaria la imagen, sólo 0 o 255
    _, thresh_img = cv2.threshold(txt_img, 128, 255, cv2.THRESH_BINARY)
    
    # Invertir imagen, cv2 jala mejor con letras blancas, fondo negro
    thresh_img = 255 - thresh_img

    # Encontrar los contornos de la imagen
    cnts, _ = cv2.findContours(thresh_img,
                               mode=cv2.RETR_EXTERNAL,
                               method=cv2.CHAIN_APPROX_SIMPLE)

    # Sacar el cuadro que encierra cada contorno
    cajas = [cv2.boundingRect(c) for c in cnts]

    # Dimensiones máximas y promedio de letra
    w_max = max((c[2] for c in cajas))
    h_max = max((c[3] for c in cajas))
    w_prom = sum((c[2] for c in cajas)) / len(cajas)
    h_prom = sum((c[3] for c in cajas)) / len(cajas)

    # Eliminar contornos muy pequeños (puntos, ruido, manchas)
    cajas = list(filter(lambda c: c[3] > 0.3*h_prom, cajas))

    # Organizar cajas de arriba a abajo
    cajas = sorted(cajas, key=lambda x: x[1])

    # Organizar cajas por renglones

    # Se inicia nuevo renglón cuando la siguiente caja tiene un
    # cambio de altura mayor a la altura promedio de una letra
    renglones = [[]]
    y = cajas[0][3] # Altura de la primera caja
    for caja in cajas[1:]: # Iterar a partir de la segunda caja
        y_prev = y
        y = caja[1]
        if (y - y_prev) > h_prom:
            renglones.append([]) #Iniciar nuevo renglón
        renglones[-1].append(caja) #Agregar en el último renglón

    # Organizar renglones de izquierda a derecha
    for i, renglon in enumerate(renglones):
        renglones[i] = sorted(renglon, key=lambda x: x[0])

    # Medir distancia promedio entre letras para identificar espacios
    spc_prom = 0
    for reng in renglones:
        n = len(reng)
        spc = (reng[i+1][0] - (reng[i][0]+reng[i][2]) for i in range(n-1))
        spc_prom += sum(spc)/(n-1)
    spc_prom /= len(renglones)

    # Construir cadena de resultado caracter por caracter
    texto = ""
    for renglon in renglones:
        x, w = (0, 0) # Primer valor para medir distancia entre letras
        for caja in renglon:
            # Añadir espacio entre letras más separadas que el promedio
            prev_x = x + w # Distancia entre letra anterior y actual
            x, y, w, h = caja
            if (x - prev_x) > 1.5*spc_prom:
                texto += " "

            # Recortar la letra de la imagen
            img_letra = np.copy(txt_img[y:y+h, x:x+w])

            # Estandarizar tamaño antes de enviar a modelo
            # Se añaden ceros hasta ser del tamaño de la letra más grande
            img_letra = np.pad(img_letra, ((0, h_max-h), (0, w_max-w)))

            # Identificar el char en la imagen
            letra = modelo(img_letra) # Aquí va el instar o el modelo que sea
            texto += letra
        texto += "\n"

    return texto, renglones


def modelo(img_letra):
    """
    Función que reciba una imagen de una letra y devuelva
    el caracter al que corresponda.

    Aquí va el modelo instar o el modelo que usemos.
    """
    return "a"



# Importar la imagen en escala de grises
txt_img = cv2.imread('datos/poema.bmp', 0)

# Producir las dimensiones y el texto contenido
texto, cajas = leer_imagen(txt_img, modelo)

print(texto)
# Ahora mismo nomás sirve para ver que sí se
# estén acomodando bien las palabras y renglones


# Dibujar algunas de las letras recortadas
letras = []
for renglon in cajas:
    for caja in renglon:
        x, y, w, h = caja
        letra = txt_img[y:y+h, x:x+w]
        letras.append(letra)

plt.figure(figsize=(20, 20))
for i in range(64):
    plt.subplot(8, 8, 1+i)
    plt.imshow(letras[i], cmap='gray')
    plt.axis('off')

plt.show()