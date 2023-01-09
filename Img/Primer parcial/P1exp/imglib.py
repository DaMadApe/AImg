import numpy as np
import cv2

"""
Archivo de funciones redactadas para manipular
imágenes, incluidos los métodos de mejora
utilizados. 
"""

def img_as_uint8(img_in):
    # Convierte imagen a np.array con rango 0-255
    scale = 255/np.max(img_in)
    img_out = np.array(img_in)
    img_out = np.rint(scale*img_out)
    img_out = np.uint8(img_out)
    return img_out


def trans_v(img, trans):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_eq = trans(np.copy(img_hsv[:,:,2]))
    img_hsv[:,:,2] = v_eq
    img_trans = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_trans


def histograma(img_in, res=255):
    # Histograma normalizado de imagen de un canal
    scale = res/np.max(img_in)
    img = np.rint(scale*img_in)
    hist = np.zeros(res+1)
    
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            idx = np.uint8(img[i][j])
            hist[idx] += 1
    return hist/sum(hist)


def acumulado(xs):
    # Acumular un arreglo en otro de igual longitud
    return [sum(xs[0:i+1]) for i in range(len(xs))]
    

def hist_stretch(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return (img-img_min) * 255/(img_max-img_min)


def ecu_hist(img, res=255):
    """
    Ecualización de histograma de un sólo canal
    res: Subdivisiones del histograma
    """
    hist = histograma(img, res=res)
    acu = acumulado(hist)
    max_val = np.max(img)
    scale = res / max_val
    img_eq = np.zeros(np.shape(img))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            idx = int(scale*img[i][j])
            img_eq[i, j] = np.uint8(acu[idx]*max_val)
    return img_eq


def frankle_mccann(img, iter=4):
    if len(img.shape)==2:
        img=img[...,None]
    ret=np.zeros(img.shape,dtype='uint8')
    def update_OP(x,y):
        nonlocal OP
        IP=OP.copy()
        if x>0 and y==0:
            IP[:-x,:]=OP[x:,:]+R[:-x,:]-R[x:,:]
        if x==0 and y>0:
            IP[:,y:]=OP[:,:-y]+R[:,y:]-R[:,:-y]
        if x<0 and y==0:
            IP[-x:,:]=OP[:x,:]+R[-x:,:]-R[:x,:]
        if x==0 and y<0:
            IP[:,:y]=OP[:,-y:]+R[:,:y]-R[:,-y:]
        IP[IP>maximum]=maximum
        OP=(OP+IP)/2
    for i in range(img.shape[-1]):
        R=np.log(img[...,i].astype('double')+1)
        maximum=np.max(R)
        OP=maximum*np.ones(R.shape)
        S=2**(int(np.log2(np.min(R.shape))-1))
        while abs(S)>=1: #iterations is slow
            for k in range(iter):
                update_OP(S,0)
                update_OP(0,S)
            S=int(-S/2)
        OP=np.exp(OP)
        mmin=np.min(OP)
        mmax=np.max(OP)
        ret[...,i]=(OP-mmin)/(mmax-mmin)*255
    return ret.squeeze()


def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_eq = trans_v(img, clahe.apply)
    return img_eq
    

def ecu_hist(img):
    return trans_v(img, cv2.equalizeHist)


def retinex_FM(img):
    return trans_v(img, frankle_mccann)