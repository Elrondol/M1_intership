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
from PIL import Image
import glob

def build_filename(path,day):
    '''Cette focntion permet d'écrire automatiquement le chemin d'accès au fichier pour une date et un jour donné'''
    if day >= 100:
        filename = f'{path}/day_{day}.h5'
    elif day<100 and day >=10 :
        filename = f'{path}/day_0{day}.h5'
    elif day < 10:
        filename = f'{path}/day_00{day}.h5'
    return filename
    
    
def data_exist(file, station,loc):
    '''Cette fonction permet de déterminer si dans un fichier il y a bien des données concernant la station choisie
    La focntion prend en compte le fait qu'il puisse n'y avoir aucune donnée du network XN (donc aucune station du réseau n'enregistre ce jour là)
    Et ça vérifie la présence du station name au sein des keys du fichier'''
    exist = False
    #verifie que y'a bien des enregistrement dans 
    if list(file.keys()).count('XN')==1: #s'il  y a bien XN dans la liste des network du fichier
        if list(file['XN'].keys()).count(f'{station}.{loc}')>0:
            exist = True
    return exist

def get_station_list(path):
    station_list = pd.read_csv(path, header=None, sep=' ', usecols=[8])
    station_list = list(station_list.values)
    station_list_clean = []
    for i in range(len(station_list)):
        station_list_clean.append(station_list[i][0])
    return station_list_clean

def date_to_day(starttime,endtime):
    '''Calcule le nombre de jours depuis le 24 juin ce qui sert ensuite à savoir dans quels fichier récupérer les données'''
    beginning = UTCDateTime('2001-06-24 00:00:00')
    daystart = int((starttime-beginning)/86400)   #il return la différence en seconde et on divise par 86400 pour avoir l'équivalent en jour 
    dayend = int((endtime-beginning)/86400)
    return daystart, dayend


def make_gif(station,duration):
    frames = [Image.open(image) for image in glob.glob(f"PSDS/{station}/*.png")]
    frame_one = frames[0]
    frame_one.save(f"PSDS/{station}/animation.gif", format="GIF", append_images=frames,
               save_all=True, duration=duration, loop=0)

def get_trace(network,station,location,year,day, dayttype='raw',cmp='Z',processed=False, fs=10):
    '''Faire une fonction permettant de récup automatiquement une trace (qui build le nom, le path etc avec comme indication la trace le jour l'année etc )'''
    if processed==True:
        path = f'data_parkfield/data_{fs}.0hz_cooked/daily/glob/{year}'
    else:
        path = f'data_parkfield/data_{fs}.0hz/daily/glob/{year}'
    
    if dayttype == 'raw':
        filename = build_filename(path, day)
    else:
        day = day+175 #on rajoute 175 parce que ça commence à day 175 pas à day 0 
        if day > 365: #dans le cas où l'on indique comme jour le nombre de jour depuis le début de la mission si on est en 2002 alors on retire 365
            day = day-365 
        filename = build_filename(path, day)
    file = h5py.File(filename,'r')
    tr = file[f'{network}'][f'{station}.{location}'][cmp][:]
    return tr
    
def merge_traces(network,station,location,channel,starttime,endtime, processed=False,fs=10):
    '''Version généralisée qui permet de changer la date de début et de fin '''
    availability_matrix = np.loadtxt('availability_matrix') #il charge depuis le fichier, donc plus besoin de compiler la cellule du dessus
    station_list_clean = get_station_list('data_parkfield/stations.txt')
    index_station = station_list_clean.index(station)
        
    #setup les infos pour tr et tr full
    tr_full = obs.Trace()
    tr_full.stats.sampling_rate = fs
    tr_full.stats.network = network
    tr_full.stats.station = station
    tr_full.stats.location = '' #trcage parce qu'on use '00' pour le fichier sauf qu'en vrai c'est ''
    tr_full.stats.channel = channel
    tr = tr_full.copy()
    
    #on doit déterminer le jour de fichier de début et de fin + il faudra q
    daystart, dayend = date_to_day(starttime, endtime) #il return les days par rapport au dbéut enregistrement  donc faudra rajouter 175 pour le fichier à build
    #dans le cas où le daystart n'a pas de données, le risque est que ce ne soit pas interprété comme un gap par la fonction ppsd, il faut par conséquent recalculer le jour où commencer
    #et la date qui lui est alors associée
    if availability_matrix[index_station,daystart]==0:#si ça commence alors qu'il n'y a pas de données
        while availability_matrix[index_station,daystart]==0:
            daystart +=1 #on increase le jour de départ jusqu'à ne plus commencer par un gap
            starttime += 86400 #on recalcule alors la date de début en accord
    
    day_list_matrix = np.arange(daystart,dayend+1) #direct les indices à chercher dans l'availability matrix du coup puisque c'est par rapport à début enregistrement
    
    tr_full.stats.starttime = starttime #on peut maitnenant donnerun début à la trace full
    
    print('Loading data...')
    for day in tqdm(day_list_matrix):
        if availability_matrix[index_station,day]!=0: #donc s'il y a de la donnée ce jour là, focntionne pour 2001 comme 2002 , du coup si y'a plus de donnée la trace sera plus courte que prévu 
            tr.stats.starttime = starttime+(day-daystart)*86400 #il update le start time à chaque itération à partir du starttime donné initialement 
            #maintenant il faut ajuster le day pour chacune des années,puisque le day qu'on utilise c'est par rapport au daystart
            if day <= 190:
                year = 2001
                day_corr = day+175 #175 car on a des données à partir du 24 juin la date de début d'enregistrement
            else: #dans ce cas on est en 2002 donc retranche une année
                year = 2002
                day_corr = day+175-365 
            tr.data = np.float64(get_trace(network,station,location,year,day_corr,cmp=channel[2], processed=processed, fs=fs))
            tr_full += tr 
    return tr_full
    
def plot_psd(network,station,location,channel,starttime,endtime,outname='', processed=False, fs=10):
    tr = merge_traces(network,station,location,channel,starttime,endtime, processed=processed, fs=fs)
    #tr = tr_full
    st = obs.Stream()
    st.append(tr)
    #création d'une réponse intrumentale plate 
    paz = {'gain':1., 'sensitivity': 1., 'poles': [1-1j], 'zeros': [1-1j]}
    #et maintenant on use obspy
    print('Creating the PPSD...')
    ppsd = PPSD(tr.stats,metadata=paz, ppsd_length=3600, skip_on_gaps=True)  #ppsd sur une journée entière donc devrait faire 479 bins, sauf qu'une bonne partie du lot y'a pas de data =0
    ppsd.add(tr)
    print('Plotting the PPSD...')
    if outname == '':
        ppsd.plot()
    else:
        ppsd.plot(filename=outname)
