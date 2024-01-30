import numpy as np
import pandas as pd
import scipy.signal as sp

def compute_distance(xs,ys,xr,yr):
    return np.sqrt((xs-xr)**2+(ys-yr)**2)    

def generate_delayed_pulse(plen,pwidth,d,c,t): #la distance et la vitesse du pulese conditionne le dt qui devra être appliqué 
    '''Fonction permettant de créer un tableau contenant les divers pulses delayed des différents receivers pour une source donnée'''
    wavelet = sp.ricker(plen,pwidth)
    
    tpulse = d/c #calcule les différents temps d'arrivée du pulse au lvl des différents receivers
    tpulse = np.round(tpulse, decimals=2) #grace au round on est sûr qu'il arrive à trouver un indice pour le tpsule dans t mais doit être changé selon le pas
    
    dt = t[1]-t[0]
    
    #on sait à quel temps on doit avoir le pulse, donc à quel temps doit faire la convolution
    
    dirac = np.zeros((len(tpulse),len(t))) # nombre de receivers en ligne et durée du signal en colonnes
    trace = np.zeros((len(tpulse),len(t)))
    
    #on boucle sur l'ensemble des stations et pour chaque station on met le pulse au temps associé
    for i in range(len(tpulse)):
        #dirac[i,np.where(t==tpulse[i])] = 1 #pour une certaine raison ça ne focntionne pas tout le temps...  
        dirac[i,int((1/dt)*tpulse[i])] = 1
        trace[i,:] = np.convolve(dirac[i,:],wavelet,mode='same')
    return trace

def compute_crosscorr(trace):
    return np.correlate(trace[0,:],trace[1,:], mode='full')
    
def generate_sources(xmin,xmax,ymin,ymax,number):
    xs = np.random.randint(low=xmin,high=xmax,size=number)
    ys = np.random.randint(low=ymin,high=ymax,size=number)
    return xs,ys


def find_nrj(amplitude,tau):
    '''lui fournir le bon axe de freq avec le bon fftshifted, il return les indexs où il y a de l'énergie, ces idnexs sont après utilisables
    pour calculer la pente et alors trouver le delay
    fonctionne que si l'énergie est concentrée dans une gamme de fréquences seulement'''
    amp_max = max(amplitude)
    ind = np.where(amplitude>tau*amp_max)
    return ind

def compute_delay_fourier(t,signal1,signal2,tau):
    ''''''
    S1 = np.fft.fft(signal1)
    S2 = np.fft.fft(signal2)
    CORR = S1*np.conjugate(S2)
    AMPLITUDE  = np.fft.fftshift(np.abs(CORR)) 
    PHASE = np.fft.fftshift(np.angle(CORR))
    PHASE_un = np.unwrap(PHASE)
    dt = t[1]-t[0]
    fs  = 1/dt
    T = dt*len(t)
    f = np.arange(-fs/2,fs/2,1/T)
    lag = np.arange(-max(t),max(t)+dt,dt)
    ind = find_nrj(AMPLITUDE,tau)
    f_cut = f[ind]
    phase_cut = PHASE_un[ind]
    a,b = np.polyfit(f_cut,phase_cut,1)
    delay = a/(2*np.pi)
    return delay
