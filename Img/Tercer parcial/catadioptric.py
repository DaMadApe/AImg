import numpy as np
import matplotlib.pyplot as plt
from skimage import io, data, color, transform

#path = 'datos/catadiop2.png'
path = 'datos/catadiop3.jpg'
img = io.imread(path)
img = color.rgb2gray(img)
h, w = img.shape

pasos = h//2 #360

recon = np.zeros((w//2, pasos))

beta_lim = np.arctan(h/(2*w))
for i in range(pasos):
    img_rot = transform.rotate(img, 360*i/pasos)
    recon[:, i] = img_rot[h//2, :w//2]

plt.imshow(recon, cmap='gray')
plt.show()