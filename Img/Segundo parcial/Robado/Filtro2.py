from skimage import io, data,morphology,feature,transform,img_as_bool
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
gris=io.imread('Image_Noise_5.tif')
plt.figure()
plt.imshow(gris,cmap='gray')
fourier1=np.fft.fft2(gris)
plt.figure()
plt.imshow(np.log(np.abs(np.fft.fftshift(fourier1))),cmap='gray')
filtro1=io.imread('filtro_linea.bmp')
filtro=img_as_bool(transform.resize(filtro1,(gris.shape[0],gris.shape[1])))*1
convolucion=np.fft.ifft2(np.fft.ifftshift((np.fft.fftshift(fourier1))*(np.invert(filtro))))
plt.figure()
plt.imshow((np.abs(convolucion)),cmap='gray')


