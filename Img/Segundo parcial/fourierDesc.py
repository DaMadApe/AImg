import pyefd
from skimage import data, measure, filters, color
import matplotlib.pyplot as plt

img = data.coins()
img_g = color.rgb2gray(img)
img_th = img_g > filters.threshold_otsu(img_g)

contours = measure.find_contours(img_th, )

fig, ax = plt.subplots()
ax.imshow(img_th, cmap='gray')
ax.plot(contours[:,0], contours[:,1])

plt.show()