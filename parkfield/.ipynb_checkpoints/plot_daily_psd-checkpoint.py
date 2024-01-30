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
import os
from parkfield_functions import *

#véritable calcul de ppsd, il faudra ensuite paralléliser cette opération  pour ça refaire une fonction qui en plus de calculer ppsd ne demande que pour argument la station, puisque c'est la seule chose qui change en

availability_matrix = np.loadtxt('availability_matrix')
station_list_clean = get_station_list('data_parkfield/stations.txt')

for station in station_list_clean:
    start = UTCDateTime('2001-06-24 00:00:00')    
    end_true = UTCDateTime('2002-10-15 00:00:00')
    index_station = station_list_clean.index(station)    
    sum_data = sum(availability_matrix[index_station,:])
    if sum_data >0:
        os.mkdir(f'PSDS/{station}')
        number=0 #sert à nommer la figure
        while start-end_true<0:   
            daystart,dayend = date_to_day(start,start)     #on trouve le jour associé aux dates données, on utilise  seulement le daystart qui nous dit si on a de la donnée pour cet indice 
            if availability_matrix[index_station,daystart] ==1:    #on doit vérifier que des data existe avant d'essayer de faire la PSD 
                outname = f'PSDS/{station}/day_{number:003}.png' #attention la date du nom de fichier est pas forcément les dates  de bout en bout si y'avait gap 
                plot_psd('XN',station,'00','BHZ',start,start,3600,outname) #en fait pour qu'il utilise qu'un jour il faut donner le même start et end 
                number+=1 
            start += 86400  #on incrémente de 1 jour  
