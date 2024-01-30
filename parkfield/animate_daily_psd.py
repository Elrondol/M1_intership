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
from PIL import Image
import glob

#variable à changer 

duration = 30

####
availability_matrix = np.loadtxt('availability_matrix')
station_list_clean = get_station_list('data_parkfield/stations.txt')

for station in station_list_clean:
    index_station = station_list_clean.index(station)
    sum_data = sum(availability_matrix[index_station,:]) #calcule la somme des données dispo sur jour, si = 0 ça veut dire que y'a de la donnée sur aucun jour, donc station a rien enregistré
    if sum_data >0:
        make_gif(station, duration) 
        
