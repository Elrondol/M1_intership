import obspy as obs
from obspy.clients.fdsn.client import Client
from obspy.signal import PPSD
from obspy import UTCDateTime
from obspy.core.trace import Stats
from obspy import Stream
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from parkfield_functions import *

#la station


## LEGACY  PAS UP TO DATE !!!!!!!!!!!!!

#####################


#######################


###################


#################


station = 'AHAB'


#des variables copiées collées du notebook

path2001 = "data_parkfield/data_10.0hz/daily/glob/2001/"
path2002 = "data_parkfield/data_10.0hz/daily/glob/2002/"

#va falloir faire la liste des stations à patir du stations.txt
station_list = pd.read_csv('data_parkfield/stations.txt', header=None, sep=' ', usecols=[8])
station_list = list(station_list.values)

station_list_clean = []

for i in range(len(station_list)):
    station_list_clean.append(station_list[i][0])
#c'est bon on a une station list clean maintenant


#maintenant on doit s'occuper des jours 

years_list = [2001,2002]

daymin2001 = 175   #24 juin 2001
daymax2001 = 365
daymin2002 = 1
daymax2002 = 288 # 15 octobre 2002  

day_list2001 = np.arange(daymin2001,daymax2001+1)
day_list2002 = np.arange(daymin2002,daymax2002+1)

day_list = np.array((list(day_list2001)+list(day_list2002))) #donc ça recommence à 1 après 365, et on connait la durée dans chaque année



#véritable calcul de ppsd, il faudra ensuite paralléliser cette opération  pour ça refaire une fonction qui en plus de calculer ppsd ne demande que pour argument la station, puisque c'est la seule chose qui change en

outname = f'PSD-{station}.pdf'
plot_psd('XN',station,'00','BHZ',station_list_clean,day_list2001,day_list2002,years_list,outname)







