import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
from skimage import (io, data, exposure, util,
                     filters, morphology, transform)

img = data.camera()

perfil = np.sum(img, axis=0)
rep = matlib.repmat(perfil, 512, 1)

perfil2 = np.sum(transform.rotate(img, 90), axis=0)
rep2 = matlib.repmat(perfil2, 512, 1)

final = rep + rep2

plt.figure()
plt.imshow(img, cmap='gray')

plt.figure()
plt.plot(perfil)
plt.figure()
plt.imshow(rep, cmap='gray')
plt.figure()
plt.imshow(rep2, cmap='gray')
plt.figure()
plt.imshow(final, cmap='gray')

plt.show()