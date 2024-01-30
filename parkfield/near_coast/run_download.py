
import numpy as np
from obspy.clients.fdsn.client import Client
from obspy.core.utcdatetime import UTCDateTime
from typing import List
from time import time
import matplotlib.pyplot as plt
from obspy.signal.spectral_estimation import PPSD
import os
from obspy.geodetics import gps2dist_azimuth
from matplotlib.colorbar import ColorbarBase
from tqdm import tqdm, trange


from basic_functions import *


def get_daily_amplitudes(net_list, sta_list,day, freqs):
    amplitude_list = np.zeros((len(sta_list),1))
    for i in range(len(sta_list)): #on gère l'ensemble des traces en rstirant al réponse + en les filtrant
        try:
            response = client.get_stations(network=net_list[0],station=sta_list[i],location='*',channel='HHZ,EHZ,BHZ',starttime=day,endtime=day,level='response')
            st = client.get_waveforms(network=net_list[0], station=sta_list[i], location='*', channel='HHZ,EHZ,BHZ',starttime=day, endtime=day+86400, attach_response=True)
            ppsd = PPSD(st[0].stats,metadata=response, ppsd_length=3600, skip_on_gaps=True)
            ppsd.add(st[0])
            try:
                os.mkdir(sta_list[i]) #fait dossier du nom de la station 
            except:
                pass
            ppsd.plot(filename=f'{sta_list[i]}/PPSD_{day.year:02}-{day.month:02}-{day.day:02}.png')
            st[0].remove_response(output='VEL')
            st[0].filter('bandpass', freqmin=freqs[0], freqmax=freqs[1])
            amplitude_list[i] = np.std(st[0].data) #calcule l'amplitude moyenne de la trace en valeur absolue
        except: #si pas pu télécharger les données alors il faut combler le trou, on comble avec un np.nan ! 
            amplitude_list[i] = np.nan
    return amplitude_list


def write_utcdatetime_list_to_file(file_path: str, utcdatetime_list: List[UTCDateTime]):
    with open(file_path, 'w') as file:
        for utcdt in utcdatetime_list:
            file.write(f"{utcdt.timestamp}\n")

def read_utcdatetime_list_from_file(file_path: str) -> List[UTCDateTime]:
    utcdatetime_list = []
    with open(file_path, 'r') as file:
        for line in file:
            timestamp_str = line.strip()
            utcdt = UTCDateTime(float(timestamp_str))
            utcdatetime_list.append(utcdt)
    return utcdatetime_list


def compute_amplitudes(net_list, sta_list, sday, eday, freqs):
    number_of_days = int((eday-sday)/86400) #calcule du nombre de jours  et on va itérer sur le nb de jour pour cjhanger le day 
    file_path = "utcdatetime_list.txt"
    try:
        utcdt_list = read_utcdatetime_list_from_file(file_path) #on lis la liste des jours déjà callculés  pour savoir si on doit ou pas get amplitudes pour ce jour! 
    except:
        write_utcdatetime_list_to_file(file_path, []) #on lui fait écrire une liste vide si la liste n'existait pas encore
        utcdt_list = read_utcdatetime_list_from_file(file_path)
        amplitudes_list = np.empty((len(sta_list),0))
        np.save('amplitudes_list.npy',amplitudes_list)
    
    for i in trange(number_of_days):
        day = sday + i*86400  #calcule le jour
        already_downloaded = day in utcdt_list
        if already_downloaded == False: #donc si on a pas encore dl le jour
            amplitudes_list = np.load('amplitudes_list.npy') #on load aussi l'array avec les amplitudes; à chaque itération on rajoutera au fichier l'amplitude des diverses station pour le jour donné
            utcdt_list = read_utcdatetime_list_from_file(file_path) #on le relis car à chaque itération il est updaté d!!
            #on chope à présent les amplitudes des diverses stations 
            amplitudes = get_daily_amplitudes(net_list=net_list,sta_list=sta_list,day=day,freqs=freqs) 
            amplitudes_list = np.hstack((amplitudes_list,amplitudes)) #on ajoute l'maplitude à la liste 
            utcdt_list.append(day) #on rjaoute à la liste de time stamps le jour donné 
            #à présent il nous faut réécrire les fichiers !!!!
            np.save('amplitudes_list.npy',amplitudes_list)
            write_utcdatetime_list_to_file(file_path, utcdt_list)

            


client = Client('NCEDC')  
net_list = ['NC']
sta_list = ['JEL','PTQ','JPL','PPB','PML','HER','PAP','BAP','PHC','PCB','PABB',
            'BPO', 'PHS', 'PAN','PADB','PSA','BCW','PPT','PLO','PBW','PMP','PHR','PJU','PAR','PDR','PCA','PHB','PWM']

sday = UTCDateTime('2001-10-02 00:00:00')
eday = UTCDateTime('2002-06-04 00:00:00')


freqs = [0.83,1.32]


compute_amplitudes(net_list=net_list, sta_list=sta_list, sday=sday, eday=eday,freqs=freqs) #devient inutile à partir du moment où on a déjà téléchargé tous les jours tbh ! 


#ON CHERCHE LES BAZ DES DIVERSES STATIONS  !!! 
baz_station = np.zeros(len(sta_list))
lat_station = np.zeros(len(sta_list))
lon_station = np.zeros(len(sta_list))

lon_bary, lat_bary, height_bary = compute_barycenter('stations_rm_nobad.txt')
for i, station in enumerate(sta_list):
    net = client.get_stations(network=net_list[0], station=station,location='*',channel='*', level='station',
                         starttime=UTCDateTime('2001-10-02'), endtime=UTCDateTime('2002-06-04')) 
    sta = net[0][0]
    lat_sta, lon_sta = sta.latitude, sta.longitude
    baz_station[i] = gps2dist_azimuth(lat_bary, lon_bary, lat_sta, lon_sta)[1]
    lat_station[i] = lat_sta
    lon_station[i] = lon_sta
    
np.save('baz_station.npy',baz_station)
np.save('lat_station.npy', lat_station)
np.save('lon_station.npy', lon_station)
