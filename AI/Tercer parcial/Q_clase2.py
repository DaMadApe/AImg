import numpy as np
import matplotlib.pyplot as plt
import random
# from drawnow import drawnow

def PosicionesIniciales(L):
    for pos in range(len( L )):
        for ind in range( len( L[pos] ) ):
            if L[pos][ind] == 80:
                [FilIni, ColIn] = [pos,ind]
            elif L[pos][ind] == 100:
                [FilMeta, ColMeta] = [pos,ind]
                return [FilIni, ColIn],[FilMeta, ColMeta]

def GiraIzquierda(D):
    if D == 0: # Arriba
        D1 = 1
    elif D == 1: # Izquierda
        D1 = 3
    elif D == 2: # Derecha
        D1 = 0
    elif D == 3: # Abajo
        D1 = 2
    return D1

def GiraDerecha(Direccion):
    if Direccion == 0:
        Direccion = 2
    elif Direccion == 1:
        Direccion = 0
    elif Direccion == 2:
        Direccion = 3
    elif Direccion == 3:
        Direccion = 1
    return Direccion

def Vecindario(fil,col,Direccion):
    global val
    valido = 0
    if Direccion == 0:
        if fil - 1 >= 0 and fil - 1 <= len(Lab):
            val = Lab[fil - 1][col]
            valido = 1
    elif Direccion == 1:
        if col - 1 >= 0 and col - 1 <= len(Lab):
            val = Lab[fil][col - 1]
            valido = 1
    elif Direccion == 2:
        if col + 1 >= 0 and col + 1 <= len(Lab):
            val = Lab[fil][col + 1]
            valido = 1
    elif Direccion == 3:
        if fil + 1 >= 0 and fil + 1 <= len(Lab):
            val = Lab[fil + 1][col]
            valido = 1
    return val,valido


def Avanza(fil,col,Direccion):
    global TempLab
    global val
    [val,valido] = Vecindario(fil,col,Direccion)
    if valido == 1:
        if val > 0:
            oldfil = fil; oldCol = col;
    if Direccion == 0: # Arriba
        fil = fil - 1
    elif Direccion == 1: # Izquierda
        col = col - 1
    elif Direccion == 2: # Derecha
        col = col + 1
    elif Direccion == 3: # Abajo
        fil = fil + 1
        status = 1
    if val == 100:
        status = 3
        print('Llegue')
        TempLab[oldfil][oldCol] = 50
        TempLab[fil][col] = 80
    elif val == 0:
        status = 2 # Obstaculo
    else:
        status = 2 # Obstaculo
    return fil,col,status

def Dibujar():
    plt.imshow(TempLab,extent = [0, 1, 0, 1])

Lab = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 80, 50, 50, 50, 50, 0, 50, 0],
[0, 50, 0, 0, 0, 50, 0, 50, 0],
[0, 50, 0, 50, 0, 50, 50, 50, 0],
[0, 0, 0, 50, 0, 0, 0, 50, 0],
[0, 50, 50, 50, 0, 50, 50, 50, 0],
[0, 50, 0, 0, 0, 50, 0, 0, 0],
[0, 50, 50, 50, 50, 50, 50,100, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0]]


[FilIn, ColIn],[FilMeta, ColMeta] = PosicionesIniciales(Lab)
print('Inicio: ',[FilIn, ColIn],'\t',' Meta: ',[FilMeta, ColMeta])

TempLab = Lab;
TempLab[FilIn][ColIn] = 50
Desplega = 1
Itera = 50
Direccion = 1


Q = np.zeros([len(Lab),len(Lab[0]), 4])
Meta = 3
Obstaculo = 2


alpha = 0.8
gamma = 0.5

Cuentapasos = []
for i in range(Itera):
    TempLab[FilMeta][ColMeta] = 100
    fil = FilIn
    col = ColIn
    status = -1
    Movimientos = 0
    Direccion = 1
    cont = 0

while status != Meta:
    prefil = fil
    precol = col
    val = np.max( Q[fil,col,:] )
    ind = np.where( Q[fil,col,:] == val )
    if len(ind[0]) > 1:
        a = len(ind[0]) - 1
        yy = int(np.round( random.random() * a ))
        accion = ind[0][yy]
    else:
        accion = ind[0][0]
        if type(accion) is np.ndarray:
            accion = accion[0]

posible = [0,1]

while accion != Direccion:
    P = int(np.round(random.random()))
    if P == 1:
        Direccion = GiraIzquierda(Direccion)
    elif P == 0:
        Direccion = GiraDerecha(Direccion)
    Movimientos = Movimientos + 1

[fil,col,status] = Avanza(fil,col,Direccion)
Movimientos = Movimientos + 1

if status == Obstaculo:
    rewardVal = -1
elif status == Meta:
    rewardVal = 1
else:
    rewardVal = 0

[prefil,precol,accion] = ( Q[prefil,precol,accion] + alpha*(rewardVal + gamma*np.max(Q[fil,col,:])) - Q[prefil,precol,accion] )
# Muestra el laberinto después de varios pasos
if Desplega == 1:
    X = [fil,col]
    Y = [FilMeta,ColMeta]
    dist = np.abs( Y[1] - X[1] + Y[0] - X[0] )
    s = 'Distancia Manhattan: ' + str(dist)
    drawnow(Dibujar)


Cuentapasos.append(Movimientos)
print('Movimientos Realizados: ',Movimientos)
drawnow(Dibujar)

# Ploteo de Movimientos
X = np.linspace(0,Itera-1,Itera)
plt.figure()
plt.bar(Itera,Cuentapasos)
plt.show()

