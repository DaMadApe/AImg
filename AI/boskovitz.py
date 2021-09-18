import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzzy
from skimage import io, data, color, exposure, transform

# Métricas de validación de cluster
def fpc(u): # Fuzzy Partition Coefficient
    #Entre más, mejor, usar con argmax
    n = u.shape[1]
    return np.trace(u.dot(u.T)) / float(n)

# [Halkidi et al]
def pec(u): # Partition entropy coefficient
    #Entre menos, mejor, usar con argmin
    n = u.shape[1]
    a = 10
    log_a_u = np.log(u)/np.log(a)
    return -np.sum(u*log_a_u)/n

# [Gath y Geva]
def fhv(datos, u): # Fuzzy hyper volume
    #Entre menos, mejor, usar con argmin
    m = 2
    result = 0
    for cluster in u:
        fuzzy_cov = 0
        v = np.average(cluster)
        for j, mship in enumerate(cluster):
            d = np.copy(datos[:,j])-v
            fuzzy_cov += (mship**m)*d*np.transpose(d)
        fuzzy_cov /= np.sum(cluster**m)
        if np.ndim(fuzzy_cov)==1:
            hv = np.abs(fuzzy_cov)**0.5
        else:
            hv = np.linalg.det(fuzzy_cov)**0.5
        result += hv
    return result

# Imagen de trabajo
path = "datos/gris5.jpg"
img = io.imread(path)

# Bajar resolución para ahorrar tiempo
#img = transform.rescale(img, 0.25)

# Hacer conversión a gris si viene en rgb
if img.shape[2]==1:
    img_v = img
else:
    img_v = color.rgb2gray(img)

# Histograma de tonos
hist, h_centers = exposure.histogram(img_v)



# Aplanar los pixeles en un sólo vector de datos
datos = np.reshape(img_v, (1, -1))
m = 2  #Parámetro de FCM, valor genérico es 2

# Obtener el mejor valor de c para FCM
c_max = 5 #Máximo número de conjuntos que puede salir
val_metrics = np.zeros(c_max+1) #Almacenar métrica para cada valor de c que probamos

for c in range(2, c_max+1):
    fcm_result = fuzzy.cmeans(datos,
                              c=c,
                              m=m,
                              error=5e-4,
                              maxiter=16) #Tal vez menos iteraciones por rendimiento
    u = fcm_result[1]
    #Cambiar para probar otra validación de cluster
    val_metrics[c] = fpc(u)
    #val_metrics[c] = pec(u)
    #val_metrics[c] = fhv(datos, u)

# Sacar la c que tuvo la mayor métrica de validación
c = val_metrics.argmax() #Cambiar argmin-argmax según métrica de validación en uso
#c = val_metrics[2:].argmin() + 2 #Offset para evitar los ceros al inicio


# Sacar FCM con el valor de C que salió
fcm_result = fuzzy.cmeans(datos,
                          c=c,
                          m=m,
                          error=5e-4,
                          maxiter=15)

# Centros y matriz de membresías
cntr, u = fcm_result[0:2]



# Formar una imagen de cada cluster
clusters = []
for cluster in u:
    """
    Cada cluster es una fila de la matriz de membresías
    y cada valor de la fila es la membresía de cada pixel
    a este cluster
    
    reacomodamos cada fila de u para que tenga la forma
    original que tenían los pixeles antes de aplanar,
    Aprovechamos que podemos representar membresía como tonos
    y los pixeles más brillantes o cercanos a 1 son los que
    pertenecen al cluster"""
    clusters.append(np.reshape(cluster, np.shape(img_v)))

# Función de membresía de un tono x a un cluster i
# La membresía de cada punto a un cluster cualquiera
# es una función de qué tan lejos está de su centro
def mship(x, i):
    den = 0
    for k in range(c):
        den += abs(x-cntr[i])/abs(x-cntr[k])**(2/(m-1))
    return 1/den

# Membresía por conjunto
xs = np.linspace(0, 1, 100) # La escala de tonos posibles
# Calcular simultaneamente las membresías de cada cluster para graficarlas
mships = [[mship(x, i) for x in xs] for i in range(c)]



# Graficar la imagen y su histograma
fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4))
ax1[0].imshow(img_v, cmap='gray')
ax1[0].axis('off')

ax1[1].plot(h_centers, hist, color='royalblue', lw=1.5)
ax1[1].set_title('Histograma')
plt.tight_layout()
plt.savefig("histograma")

# Graficar cada cluster y respectivas funciones de membresía
fig2, ax2 = plt.subplots(2, c, figsize=(3*c, 6))
for i, cluster in enumerate(clusters):
    ax2[0, i].imshow(cluster, cmap='gray')
    ax2[0, i].axis('off')
    ax2[0, i].set_title(f"Cluster {i}")

    ax2[1,i].plot(xs, mships[i], color='royalblue')
    ax2[1,i].set_title(f"Membresía de cluster {i}")
plt.tight_layout()
plt.savefig("clusters")

# Graficar la medida de validación para cada C que probamos
fig3, ax3 = plt.subplots()
ax3.plot(val_metrics, color='r', lw=1.5)
ax3.set_title('Validación por número de cluster')
ax3.set_xlabel('c')
ax3.set_ylabel('FPC')
plt.tight_layout()
plt.savefig("val_fpc")

plt.show()