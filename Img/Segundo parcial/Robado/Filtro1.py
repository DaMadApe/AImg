import numpy as np
import matplotlib.pyplot as plt
from skimage import io, draw, transform,img_as_bool


plt.close('all')

gris = io.imread('Image_Noise_4.tif')
plt.figure()
plt.imshow(gris, cmap='gray')

fourier1 = np.fft.fftshift(np.fft.fft2(gris))
plt.figure()
plt.imshow(np.log(np.abs(fourier1)), cmap='gray')

pos = np.int32(plt.ginput(0,0))
fourier2 = img_as_bool(np.zeros((gris.shape[0], gris.shape[1])))

for i in range(len(pos)):
    rr, cc = draw.circle(pos[i,1], pos[i,0], 10, fourier2.shape)
    fourier2[rr,cc] = 1
plt.figure()
plt.imshow(fourier2, cmap='gray')
plt.show()

salida = np.fft.ifft2(np.fft.ifftshift(fourier1*np.invert(fourier2)))
plt.figure()
plt.imshow(np.abs(salida), cmap='gray')

