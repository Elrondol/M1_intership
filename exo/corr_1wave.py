import numpy as np
import pandas as pd
import scipy as sp

from functions import compute_distance,generate_delayed_pulse,compute_crosscorr,generate_sources

c = 1500 
t = np.arange(0,10,0.01) 
xs,ys = generate_sources(-2000,2000,-2000,2000,50000)
xr = np.array([100,200]) 
yr = np.array([200,-600])

corr = np.zeros(2*len(t)-1)

for i in range (len(xs)):
    d = compute_distance(xs[i],ys[i],xr,yr)
    trace = generate_delayed_pulse(50,5,d,c,t)
    corr += compute_crosscorr(trace)
    
np.savetxt('output.txt',corr)
