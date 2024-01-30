import numpy as np
import pandas as pd
import scipy as sp

from functions import compute_distance,generate_delayed_pulse,compute_crosscorr,generate_sources

c1,c2 = 1000,600 

t = np.arange(0,100,0.01)

xr = np.array([100,200])
yr = np.array([200,-600])

x0 = np.mean(xr)
y0 = np.mean(yr)   #on met les sources sur un cercle centré sur le barycentre des deux récepteurs

rayon = np.linspace(800,50000,1000)
angles = np.linspace(0,2*np.pi,500) 

xs_cercle = []
ys_cercle = []

for i in range(len(rayon)):
    xs_cercle += np.ndarray.tolist(x0+rayon[i]*np.sin(angles))
    ys_cercle += np.ndarray.tolist(y0+rayon[i]*np.cos(angles))

xs_cercle,ys_cercle = np.array(xs_cercle), np.array(ys_cercle) 


corr = np.zeros(2*len(t)-1)
lag = np.linspace(-max(t),max(t),len(corr))

lag_list = []

corrsum = np.zeros(len(corr))

for i in range (len(xs_cercle)):
    d = compute_distance(xs_cercle[i],ys_cercle[i],xr,yr)
    tracep1 = generate_delayed_pulse(50,5,d,c1,t)
    tracep2 = generate_delayed_pulse(50,5,d,c2,t)
    trace = tracep1+tracep2
    corr = compute_crosscorr(trace)
    corrsum += corr
    lag_list.append(lag[np.where(corr==max(corr))][0]) 
    

#on réorganise sous forme de kernel la lag list
lag_list = np.reshape(lag_list, (len(rayon),len(angles)))

np.savetxt('out_corrsum',corrsum)
np.savetxt('out_laglist',lag_list)
