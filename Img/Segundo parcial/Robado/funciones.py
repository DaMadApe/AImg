import numpy as np
import scipy.interpolate as interpolate

def extremo(Ima):
    
    [f,c] = Ima.shape
    # Encuentra posición de todos los pixeles blancos 
    fs,cs = np.where(Ima > 0)
    
    # Obtiene kernel horizontal y vertical
    v,h = np.meshgrid([1,0,1],[1,0,1])
    
    # Crea matriz vacía para la imagen con los extremos
    new = np.zeros((f,c))
    
    # Para todos los pixeles blancos
    for i in range(len(fs)):
        
        # Obtiene su vecindario del pixel i
        reg = Ima[fs[i] - 1:fs[i] + 2,cs[i] - 1:cs[i] + 2]
        
        #Busca pixeles en h,v y hv
        sh = np.sum(np.sum(reg*h,axis = 1) > 0) > 1
        sv = np.sum(np.sum(reg*v,axis = 0) > 0) > 1
        es = np.sum(reg*v*h) > 3
        
        # Forma el pixel i en la nueva imagen
        new[fs[i],cs[i]] = (sv + sh + es)
        
    # Obtiene la iamgen con los posibles extremos 
    new = Ima - new 
    
    # Obtiene las coordenadas de los pixeles 
    fs,cs = np.where(new == 1)
    
    # Crea un vector para lamacenar el # de vecinos
    vecinos = np.zeros(len(fs))
    
    # Radio del vecindario
    r = 5
    
    for i in range(len(fs)):
        
        # Obtiene su vecindario
        reg = Ima[fs[i] - r:fs[i] +r+1,cs[i] - r:cs[i] +r+1]    
        
        # Cuenta el número de vecinos
        vecinos[i] = np.sum(reg)
        
    # Define como inicio al que tiene menos vecinos

    if len(fs) > 0:
        i_x = fs[np.argmin(vecinos)]
        i_y = cs[np.argmin(vecinos)]
    else:
        i_x = -1
        i_y = -1
        
    return i_x,i_y


def ordena(Ima,i_x,i_y,min_len):
    stop = 0
    counter = 0
    eliminados = 0
    
    e_x = int(i_x)
    e_y = int(i_y)
    xcoord = np.empty(0)
    ycoord = np.empty(0)
    elim_x = np.empty(0)
    elim_y = np.empty(0)
    
    orden = np.array([[1,2,3],[8,0,4],[7,6,5]])
    direccion_x = np.array([-1,-1,-1,0,1,1,1,0])
    direccion_y = np.array([-1,0,1,1,1,0,-1,-1])
    
    while stop == 0:
        counter = counter + 1
        e_x = int(e_x)
        e_y = int(e_y)
        
        Ima[e_x,e_y] = 0
        xcoord = np.append(xcoord,e_x)
        ycoord = np.append(ycoord,e_y)
        
        reg = Ima[e_x - 1:e_x + 2,e_y - 1:e_y + 2]*orden
        
        ind = np.max(reg)
        
        if ind > 0:
            e_x = e_x + direccion_x[ind - 1]
            e_y = e_y + direccion_y[ind - 1]
            
            if eliminados > 0 :
                eliminados = 0
                
                elim_x = np.empty(0)
                elim_y = np.empty(0)
                
            
        else :
            
            e_x = xcoord[-1]
            e_y = ycoord[-1]
            
            eliminados += 1
            elim_x = np.append(elim_x,e_x)
            elim_y = np.append(elim_y,e_y)
            
            xcoord = xcoord[0:-1]
            ycoord = ycoord[0:-1]
            
        if eliminados > min_len:
            stop = 1
            
            if len(xcoord) > min_len:
                xcoord = np.append(xcoord,elim_x)
                ycoord = np.append(ycoord,elim_y)
    
    return Ima,xcoord,ycoord


def get_poly(xvec,yvec):
    
    # Define un número de puntos
    num_puntos = np.fix(len(xvec)*0.1).astype(int)
    
    # Si hay menos de  puntos, usa todos
    if num_puntos < 4:
        num_puntos = len(xvec) 
    
    # Define un vector de índices equidistantes
    ind = np.fix(np.linspace(0,len(xvec) - 1,num_puntos)).astype(int)
    
    #se queda sólo con los puntos seleccionados
    xvec = xvec[ind]
    yvec = yvec[ind]
    
    #Obtiene la función de interpolación en x e y
    funcx = interpolate.splrep(ind, xvec, s=0)
    funcy = interpolate.splrep(ind, yvec, s=0)
    
    # Obtiene los tados de la función de interpolación
    
    ppx = interpolate.PPoly.from_spline(funcx)
    ppy = interpolate.PPoly.from_spline(funcy)

    return [ppx,ppy]

























