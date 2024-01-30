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


starttime = UTCDateTime('2001-06-24 00:00:00')
endtime = UTCDateTime('2002-10-15 00:00:00')

os.mkdir(f'PSDS/{starttime}_{endtime}')

#véritable calcul de ppsd, il faudra ensuite paralléliser cette opération  pour ça refaire une fonction qui en plus de calculer ppsd ne demande que pour argument la station, puisque c'est la seule chose qui change en

availability_matrix = np.loadtxt('availability_matrix')
station_list_clean = get_station_list('data_parkfield/stations.txt')

for station in station_list_clean:
    index_station = station_list_clean.index(station)
    sum_data = sum(availability_matrix[index_station,:]) #calcule la somme des données dispo sur jour, si = 0 ça veut dire que y'a de la donnée sur aucun jour, donc station a rien enregistré
    if sum_data >0:
        outname = f'PSDS/{starttime}_{endtime}/PSD-{station}.pdf' #attention la date du nom de fichier est pas forcément les dates  de bout en bout si y'avait gap 
        plot_psd('XN',station,'00','BHZ',starttime,endtime,3600,outname)
