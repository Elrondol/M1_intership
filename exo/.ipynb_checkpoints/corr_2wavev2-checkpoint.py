import numpy as np
import pandas as pd
import scipy as sp
import ray
ray.init(num_cpus=4)
from functions import compute_distance,generate_delayed_pulse,compute_crosscorr,generate_sources


xr = np.array([100,200])
yr = np.array([200,-600])

x0 = np.mean(xr)
y0 = np.mean(yr)   #on met les sources sur un cercle centré sur le barycentre des deux récepteurs

t = np.arange(0,100,0.01)
lag = np.arange(-99.99,100,0.01)

rayon = np.linspace(800,50000,100)
angles = np.linspace(0,2*np.pi,50) 

@ray.remote
def compute_corr_angles(rayon):
    '''Calcule les corréaltion pour tous les angles et les somme et ce pour un rayon donné. On va ensuite devoir boucler sur les rayons pour avoir le résultat souhaité.'''
    xs_cercle = x0+rayon*np.sin(angles)
    ys_cercle = y0+rayon*np.cos(angles)
    corrsum = np.zeros(2*len(t)-1)
    for i in range(len(angles)):
        d = compute_distance(xs_cercle[i],ys_cercle[i],xr,yr)
        tracep1 = generate_delayed_pulse(50,5,d,1000,t)
        tracep2 = generate_delayed_pulse(50,5,d,600,t)
        trace = tracep1+tracep2
        corrsum += compute_crosscorr(trace)
    return corrsum

corrsum_list = ray.get([compute_corr_angles.remote(rayon[i]) for i in range(len(rayon))])

corrsum = np.sum(corrsum_list,axis=0)

np.savetxt('out_corrsumv2-1',corrsum)
