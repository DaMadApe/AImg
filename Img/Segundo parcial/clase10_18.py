from skimage import data, color
import numpy as np
import matplotlib.pyplot as plt

def pasabajos_ideal(shape, f_corte):
    fil, col = shape
    y, x = np.ogrid[-fil/2:fil/2, -col/2:col/2]
    filtro = (x**2 + y**2 <= f_corte**2)
    return filtro.astype(float)

def pasaaltos_ideal(shape, f_corte):
    fil, col = shape
    y, x = np.ogrid[-fil/2:fil/2, -col/2:col/2]
    filtro = (x**2 + y**2 >= f_corte**2)
    return filtro.astype(float)

def pasabajos_butter(shape, f_corte, N):
    fil, col = shape
    y, x = np.ogrid[-fil/2:fil/2, -col/2:col/2]
    circulo = (x**2 + y**2 <= f_corte**2).astype(float)

    filtro = 1 / (1 + (circulo/f_corte)**(2*N))
    return filtro.astype(float)

def pasaaltos_butter(shape, f_corte, N):
    fil, col = shape
    y, x = np.ogrid[-fil/2:fil/2, -col/2:col/2]
    circulo = (x**2 + y**2 >= f_corte**2)
    return filtro.astype(float)

def aplicar_filtro_frec(img, kernel):
    if kernel.shape == img.shape:
        filtro = kernel
    else:
        filtro = np.zeros(img.shape)
        filtro[:kernel.shape[0], :kernel.shape[1]] = kernel

    img_frec = np.fft.fftshift(np.fft.fft2(img))
    filtro = pasaaltos_ideal(img.shape, 100)

    # Aplicar el filtro en frecuencia
    convolucion = img_frec * filtro
    # Regresar al dominio espacial
    img_ift = np.fft.ifft2(np.fft.fftshift(convolucion))
    img_final = np.abs(img_ift)
    img_final = util.img_as_ubyte(img_final/np.max(img_final))
    return img_final




img = data.camera()
img_ft = np.fft.fftshift(np.fft.fft2(img))

filtro = pasabajos_butter(img.shape, 100, 20)
convolucion = img_ft * filtro

img_ift = np.fft.ifft2(np.fft.fftshift(convolucion))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')

plt.subplot(1,3,2)
plt.imshow(filtro, cmap='gray')

plt.subplot(1,3,3)
plt.imshow(np.abs(img_ift), cmap='gray')

plt.show()