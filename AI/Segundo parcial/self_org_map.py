from skimage import io, data, color
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

nfil = 60
ncol = 100
nras = 3
som = np.random.rand(nfil, ncol, nras)
plt.ion()
plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(som)
datos = np.random.rand(100, nras)
x = np.linspace(0, 100, ncol)
y = np.linspace(0, 60, nfil)
x, y = np.meshgrid(x, y)
epocas = 5
alpha0 = 0.5
decay = 0.05
sgm0 = 20
for t in range(epocas):
    alpha = alpha0 * np.exp(-t * decay)
    sgm = sgm0 * np.exp(-t * decay)
    ven = np.ceil(sgm*3)
    for i in range(100):
        vector = datos[i, :]
        columna = som.reshape(nfil*ncol, 3)
        d = 0
        for n in range(3):
            d = d + (vector[n]-columna[:, n])**2
            DISTAN = np.sqrt(d)
            ind = np.argmin(DISTAN)
            bmfil, bmcol = np.unravel_index(ind, [nfil, ncol])
            g = np.exp( -( ( (x-bmcol)**2) + ((y-bmfil)**2) ) / (2*sgm*sgm) )
            ffil = int( np.max( [0, bmfil-ven] ) )
            tfil = int( np.min( [bmfil+ven, nfil] ) )
            fcol = int( np.max( [0, bmcol-ven] ) )
            tcol = int( np.min( [bmcol+ven, ncol] ) )
            vecindad = som[ffil:tfil, fcol:tcol, :]
            a, b, c = vecindad.shape
            T = np.ones(vecindad.shape)
            T[:,:,0] = T[:,:,0] * vector[0]
            T[:,:,1] = T[:,:,1] * vector[1]
            T[:,:,2] = T[:,:,2] * vector[2]
            # T = np.reshape(np.tile(vector, (1, a*b)), [a, b, nras])
            G = np.ones(vecindad.shape)
            G[:,:,0] = g[ffil:tfil, fcol:tcol]
            G[:,:,1] = g[ffil:tfil, fcol:tcol]
            G[:,:,2] = g[ffil:tfil, fcol:tcol]
            # G = np.tile(g[ffil-1:tfil+2, fcol-1:tcol+2], [1, 1, 3])
            vecindad = vecindad + (alpha*G*(T-vecindad))
            som[ffil:tfil, fcol:tcol, :] = vecindad
    #plt.subplot(1, 2, 2)
    plt.imshow(som)
    plt.pause(0.05)
plt.show()
