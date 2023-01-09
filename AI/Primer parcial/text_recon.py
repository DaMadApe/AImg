import numpy as np
import cv2
import matplotlib.pyplot as plt


def leer_texto(txt_img, outsize, modelo=None, padding=True):
    """
    Detectar y recortar las letras de la imagen,
    y aplicar el modelo de lectura a cada una
    para devolver una reconstrucción del texto original,
    y un arreglo con las imágenes de las letras sólas.
    Sin modelo, devuelve sólo una cadena con la estructura
    del texto y el arreglo de las letras.
    """
    # Hacer binaria la imagen, sólo 0 o 255
    _, thresh_img = cv2.threshold(txt_img, 100, 255, cv2.THRESH_BINARY)
    
    # Invertir imagen, cv2 jala mejor con letras blancas, fondo negro
    thresh_img = 255 - thresh_img

    # Encontrar los contornos de la imagen
    cnts, _ = cv2.findContours(thresh_img,
                               mode=cv2.RETR_EXTERNAL,
                               method=cv2.CHAIN_APPROX_SIMPLE)

    # Sacar el cuadro que encierra cada contorno
    cajas = [cv2.boundingRect(c) for c in cnts]

    # Dimensiones de referencia sobre tamaño de letra
    w_prom = sum((c[2] for c in cajas)) / len(cajas)
    h_prom = sum((c[3] for c in cajas)) / len(cajas)

    # Eliminar contornos muy pequeños (puntos, ruido, manchas)
    cajas = list(filter(lambda c: c[3] > 0.3*h_prom, cajas))

    # Nuevos tamaños mínimos
    w_min = min((c[2] for c in cajas))
    h_min = min((c[3] for c in cajas))

    # Organizar cajas de arriba a abajo
    cajas = sorted(cajas, key=lambda x: x[1])

    # Organizar cajas por renglones

    # Se inicia nuevo renglón cuando la siguiente caja tiene una
    # diferencia vertical mayor a la altura promedio de una letra
    renglones = [[]]
    yb = cajas[0][1]+cajas[0][3] # Altura de la base de la primera letra
    renglones[-1].append(cajas[0])
    for caja in cajas[1:]: # Iterar a partir de la segunda caja
        yb_prev = yb
        yb = caja[1] + caja[3] # Altura de la base de la letra
        if (yb - yb_prev) > h_prom:
            renglones.append([]) # Iniciar nuevo renglón
        renglones[-1].append(caja) # Agregar en el último renglón

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
    # Incluir las letras recortadas para visualización
    imgs_letras = []

    for renglon in renglones:
        x, w = (0, 0) # Primer valor para medir distancia entre letras
        for caja in renglon:
            # Añadir espacio entre letras más separadas que el promedio
            prev_x = x + w # Distancia entre letra anterior y actual
            x, y, w, h = caja
            if (x - prev_x) > 1.5*spc_prom:
                texto += " "

            # Recortar la letra de la imagen
            img_letra = np.copy(thresh_img[y:y+h, x:x+w])

            # Agregar padding si es muy chica la letra
            if padding:
                if w < 1.2*w_min: 
                    img_letra = np.pad(img_letra, ((0,0), (0, int(w_min/2))))
                if h < 1.2*h_min: 
                    img_letra = np.pad(img_letra, ((0, int(h_min/2)), (0,0)))

            # Estandarizar tamaño de imágenes al tamaño especificado
            img_letra = cv2.resize(img_letra, outsize, interpolation=cv2.INTER_CUBIC)

            imgs_letras.append(img_letra)

            # Identificar el char en la imagen
            if modelo is not None:
                letra = modelo(img_letra) # Aquí va el instar o el modelo que sea
            else:
                letra = "a" # Prestalugar, sirve para ver la estructura de los renglones
                            # O para producir sólo el conjunto de imágenes de letras
            texto += letra
        texto += "\n"

    return texto, imgs_letras


"""
Modelo para reconocimiento de patrones
"""

def compet(x):
    a = np.argmax(x)
    return a

def modelo(img_letra, W):
    abcdef = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
              'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
              'a','b','c','d','e','f','g','h','i','j','k','l','m',
              'n','o','p','q','r','s','t','u','v','w','x','y','z']
    # Acomodar imagen como columna
    p = img_letra.reshape(-1, 1)
    # Normalizar entrada
    p = p/(np.linalg.norm(p)+0.001)
    a = compet(W.dot(p))
    letra = abcdef[a]
    return letra


# Formar el identificador de patrones

# Importar imagen con la fuente arial
arial_img = cv2.imread('datos/arial_font.jpg', 0)

# Resolución de trabajo
res = (12, 12)

# Extraer imágenes de caracteres arial
_, arial_imgs = leer_texto(arial_img, outsize=res)

# Crear los pesos del modelo identificador
W = np.zeros((len(arial_imgs), res[0]*res[1]))
for n, letra in enumerate(arial_imgs):
    # Formar el vector prototipo de cada letra
    # usando la tipografía importada como referencia
    W[n] = letra.flatten() / np.linalg.norm(letra)


"""
Aplicar el modelo
"""

# Importar la imagen en escala de grises
txt_img = cv2.imread('datos/poema.bmp', 0)

# Leer el texto contenido
texto, imgs_letras = leer_texto(txt_img, outsize=res,
                                modelo=lambda im: modelo(im, W))
print(texto)


""" 
# Visualizar caracteres como los recibe el modelo
plt.figure(figsize=(10, 4))
for i in range(len(arial_imgs)):
    plt.subplot(4, 13, 1+i)
    plt.imshow(arial_imgs[i], cmap='gray')

plt.figure(figsize=(10, 10))
for i in range(100):
    plt.subplot(10, 10, 1+i)
    plt.imshow(imgs_letras[i], cmap='gray')

plt.show()
 """