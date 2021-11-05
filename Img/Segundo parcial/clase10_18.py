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