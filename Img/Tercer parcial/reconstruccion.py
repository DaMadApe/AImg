import numpy as np
import matplotlib.pyplot as plt
"""
Hacer la visualización en voxels de la reconstrucción
producida por reconst3d.py
"""
recons = np.load('recons_craneo3d.npy')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=110, azim=60)
ax.voxels(recons)
plt.show()