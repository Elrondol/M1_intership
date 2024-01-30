import numpy as np
import pandas as pd
from pyproj import Transformer
import m_beam
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from variables import *
from tqdm import tqdm
import h5py

from sklearn.cluster import DBSCAN
import scipy.ndimage
import scipy
import matplotlib.cm as cm
from skimage.feature import blob_doh, blob_dog, blob_log
from findpeaks import findpeaks

from matplotlib.transforms import Bbox

from obspy.geodetics.base import gps2dist_azimuth
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.img_tiles import Stamen, OSM, GoogleTiles
from matplotlib_scalebar.scalebar import ScaleBar
from geopy.distance import distance, geodesic
from pyrocko import cake

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from obspy.core.utcdatetime import UTCDateTime
from scipy.signal import hilbert,find_peaks, butter, filtfilt

from typing import List
import netCDF4 as nc



def write_datetime_list_to_file(file_path: str, datetime_list: List[datetime]):
    with open(file_path, 'w') as file:
        for dt in datetime_list:
            file.write(f"{dt.isoformat()}\n")

def read_datetime_list_from_file(file_path: str) -> List[datetime]:
    datetime_list = []
    with open(file_path, 'r') as file:
        for line in file:
            dt_str = line.strip()
            dt = datetime.fromisoformat(dt_str)
            datetime_list.append(dt)
    return datetime_list

def extract_coordinates(path,sta_name=False,sta_loc=False):
    '''Fonction qui récupère en input la liste des station et qui va  extraire les coordonnées
    L'option sta_name permet de récupérer les noms des stations en plus des autres listes'''
    #laoding the station_list, cette version de station list devra être la version avec les stations non utilisées virées 
    latitudes = pd.read_csv(path, header=None, sep='    ',usecols=[4],engine='python').values
    longitudes = pd.read_csv(path, header=None, sep='    ', usecols=[5],engine='python').values
    heights = pd.read_csv(path, header=None, sep='    ', usecols=[6], engine='python').values
    latitudes_clean = []
    longitudes_clean = []
    heights_clean = []
    if sta_name==True:
        networks = pd.read_csv(path, header=None, sep='    ',usecols=[1],engine='python').values
        networks_clean = []
        stations = pd.read_csv(path, header=None, sep='    ',usecols=[2],engine='python').values
        stations_clean = []
    
    if sta_loc==True:
        locations = pd.read_csv(path, header=None, sep='    ',usecols=[3],engine='python').values
        locations_clean = []
        
    for i in range(len(latitudes)):
        latitudes_clean.append(latitudes[i][0])
        longitudes_clean.append(longitudes[i][0])
        heights_clean.append(heights[i][0])
        if sta_name==True:
            networks_clean.append(networks[i][0])
            stations_clean.append(stations[i][0])
        if sta_loc==True:
            if locations[i][0]=='--':
                locations_clean.append('00')
            else:
                locations_clean.append(locations[i][0])
        
    latitudes_clean = np.array(latitudes_clean)    
    longitudes_clean = np.array(longitudes_clean)
    heights_clean = np.array(heights_clean)

    if sta_name == True and sta_loc ==True:
        networks_clean = np.array(networks_clean)
        stations_clean = np.array(stations_clean)
        locations_clean = np.array(locations_clean)
        return longitudes_clean,latitudes_clean,heights_clean, networks_clean, stations_clean, locations_clean
    elif sta_name == True:
        networks_clean = np.array(networks_clean)
        stations_clean = np.array(stations_clean)
        return longitudes_clean,latitudes_clean,heights_clean, networks_clean, stations_clean
    elif sta_loc ==True:
        locations_clean = np.array(locations_clean)
        return longitudes_clean,latitudes_clean,heights_clean, locations_clean
    else:
        return longitudes_clean,latitudes_clean,heights_clean

def compute_barycenter(path):
    '''Fonction permettant de calculer les coordonnées du barycentre de de l'array à partir du fichier station list'''
    longitudes_clean,latitudes_clean,heights_clean = extract_coordinates(path)
    #loading the transformers to convert coordinates to     
    trans_GPS_to_XYZ = Transformer.from_crs(4979, 4978, always_xy=True)
    trans_XYZ_to_GPS = Transformer.from_crs(4978,4979,always_xy=True)
    #on a créé les convertisseurs    
    X,Y,Z = trans_GPS_to_XYZ.transform(longitudes_clean, latitudes_clean, heights_clean)
    X_bary = np.mean(X) 
    Y_bary = np.mean(Y)
    Z_bary = np.mean(Z)
    lon_bary,lat_bary,height_bary = trans_XYZ_to_GPS.transform(X_bary,Y_bary,Z_bary)
    return lon_bary, lat_bary, height_bary

def compute_time_string(index_N0, ff):
    '''Calcule l'heure devant être indiquée dans le titre; (heure de début de la window'''
    #beginning time 
    time = (index_N0/ff) 
    hours = int(time//3600)
    time_left = time-hours*3600
    minutes = int(time_left//60)
    seconds = int(time_left-60*minutes)
    return (f'{hours:002}h-{minutes:002}mn-{seconds:002}s')

def compute_velo(loc_max,U):
    x_dist = U[loc_max[1][0]]
    y_dist = U[loc_max[0][0]] 
    dist = np.sqrt((x_dist)**2+(y_dist)**2)
    return 1/dist

def compute_baz(loc_max,U):
    x_dist = U[loc_max[1][0]]
    y_dist = U[loc_max[0][0]]
    if x_dist==0 and y_dist==0: #pour éviter le NAN 
        baz = 0.
    elif x_dist >=0 and y_dist > 0 :
        baz = np.rad2deg(np.arctan(x_dist/y_dist))
    elif x_dist >0 and y_dist<=0:
        baz = np.rad2deg(np.arctan(-y_dist/x_dist))+90
    elif x_dist <=0 and y_dist <0:
        baz = np.rad2deg(np.arctan(-x_dist/-y_dist))+180
    elif x_dist <0 and y_dist >=0:
        baz = np.rad2deg(np.arctan(y_dist/-x_dist))+270
    else:
        print('bruh')
    return baz

def extract_velo_baz(year, days,veloc_range=None, baz_range=None,plot_blobs=False, clim=None, blend=False):
    if plot_blobs==True:
        #création du répertoire et création du chemin qui servira à save les figures
        if blend==True: #la seule utilité de rajouter l'argument blend à la fonction!
            folder = in_['beam_dir'][15:] + '_wblobs_wblend' 
        else:
            folder = in_['beam_dir'][15:] + '_wblobs'
        path = f'/summer/faultscan/user/parisnic/m1_internship/parkfield/BEAM_FIGS/{folder}'
        try:
            os.makedirs(path)                 
        except:
            print('Folder already created')
            
    veloc_matrix = np.zeros((len(pl_updated['fn']),len(days)))
    baz_matrix = np.zeros((len(pl_updated['fn']),len(days)))
    amplitude_matrix = np.zeros((len(pl_updated['fn']),len(days)))
    incert_veloc_matrix = np.zeros((2,len(days)))
    incert_baz_matrix = np.zeros((2,len(days)))
    
    print(f'Processing {year} data, please wait...')  
    
    for i in tqdm(range(len(days))):
        if int(days[i])!=days[i]: #si blend est true, alors il y a des floats .5 et donc cette égalité est pas respectée 
            day_first = int(days[i])
            day_second = day_first+1
            data1 = h5py.File(f"{in_['beam_dir']}/beam_{bb['compo']}_{year}_day_{day_first:03}.h5",'r') 
            data2 = h5py.File(f"{in_['beam_dir']}/beam_{bb['compo']}_{year}_day_{day_second:03}.h5",'r')
            fk1   = data1['fk'][()]
            fk2   = data2['fk'][()]
            freqs = data1['ff']['sfreq'][()]   #ça devrait pas matter si on prend depuis le 1 ou le 2
            U    = data1['bb']['U'][()] * 10**3 #doit direct utiliser U pour les calculs avec x et y!!!
        else: #si respectée 2 poss, soit c'est que c'est blend mais jour full soit c'est pas blend et donc déjà int
            if isinstance(days[i], float)==True: #si c'est in float alors on le reconverti en int
                day = int(days[i])
            else:
                day = days[i]
            data = h5py.File(f"{in_['beam_dir']}/beam_{bb['compo']}_{year}_day_{day:03}.h5",'r') #il importe le fichier 
            fk   = data['fk'][()]
            freqs = data['ff']['sfreq'][()]
            U    = data['bb']['U'][()] * 10**3
                
        #on doit d'abord vérifier qu'il  y a de la donnée ce jour là histoire d'éviter un crash! 
        availability_matrix = np.loadtxt('availability_matrix_nobad')
        
        #on doit calculer le nombre de stations, pour ça, soit c'est un jour et ez, soit c'est deux jours et alors on prend le nb minmum entre les deux
        if int(days[i])!=days[i]:
            idx_availability1 = installed_days(year, day_first)
            idx_availability2 = installed_days(year, day_second)
            sta_number1 = np.count_nonzero(availability_matrix[:,idx_availability1]==1)
            sta_number2 = np.count_nonzero(availability_matrix[:,idx_availability2]==1)
            sta_number = np.min([sta_number1,sta_number2])
            if day_first==102 or day_second==102:
                sta_number = 0 #on truque pour que 102 fasse pas planter 
        else:#on peut direct utiliser day ici, car calculé précéemment 
            idx_availability = installed_days(year, day)
            sta_number = np.count_nonzero(availability_matrix[:,idx_availability]==1)
        
        if sta_number>0 and days[i]!=102: #s'assure d'avoir de la donnée avant d'essayer de faire le beamforming
            for j in range(len(pl_updated['fn'])):
                ifreq = m_beam.make_short_freq(pl_updated['fn'][j],freqs,fw=pl_updated['fw'],nf=0)
                
                if int(days[i])!=days[i]:   #pareil, si on est sur un .5, alors on doit blend les deux jours!! 
                    Z1 = np.mean(np.abs(fk1[ifreq,:,:]),axis=0).transpose()
                    Z2 = np.mean(np.abs(fk2[ifreq,:,:]),axis=0).transpose()
                    Z  = (Z1+Z2)/2 #on fait la moyenne des beams des deux jours différents !
                else: #si int(days) = days alors c'est qu'on est sur un jour spécifique et donc on a déjà son Z grâce au fk
                    Z = np.mean(np.abs(fk[ifreq,:,:]),axis=0).transpose() #on a supprimé l'option pour faire la déconvol car travaille qu'en bartlett
                #on passe en échelle db si besoin !
                if pl_updated['db']:
                    Z = 20 * np.log10(Z)
                #maintenant que l'on a correctement traité Z, il a la même alors que sur les plots de beam, et donc les valeurs extraites sont les mêmes que celles
                #attendues à partir des panneaux de beam !
                loc, veloc_matrix[j,i],baz_matrix[j,i],amplitude_matrix[j,i],incert_veloc, incert_baz = find_local_max(Z,U,veloc_range,
                                                                                                                    baz_range,plot_blobs=plot_blobs, clim=clim)
                #on retient aussi l'amplitude du pic trouvé, afin de savoir si c'est un vrai pic ou bien si c'est pas fiable 
                #mais peut y avoir pb si le signal est saturé car apparait forte amplitud ealors que juste pb dans la trace ...
                incert_veloc_matrix[0,i] = incert_veloc[0]
                incert_veloc_matrix[1,i] = incert_veloc[1]
                incert_baz_matrix[0,i] = incert_baz[0]
                incert_baz_matrix[1,i] = incert_baz[1]
                if plot_blobs==True: #on peut à présent sauvegarder !! le fichier dans son folder et sur lequel apparait le jour Julian 
                    plt.title(f'Day {days[i]} of {year}')
                    plt.savefig(f'{path}/beam_{compo_to_plot}_{year}_day_{days[i]:05}.png',bbox_inches='tight', dpi=300)
        else:
            for j in range(len(pl_updated['fn'])):
                loc, veloc_matrix[j,i],baz_matrix[j,i],amplitude_matrix[j,i],incert_baz_matrix[0,i],incert_baz_matrix[1,i] = ([0],[0]), 999,999,0,0,0 
                incert_veloc_matrix[0,i],incert_veloc_matrix[1,i]=0,0 
                
    return veloc_matrix, baz_matrix, amplitude_matrix, incert_veloc_matrix, incert_baz_matrix


def installed_days(year,day):
    '''Même chose que days since install, mais qui prend en arguments l'année et le jour de l'année au lieu de la date'''
    if year ==2001:
        days_since_install = day-175
    elif year==2002:
        days_since_install = day+365-175
    return days_since_install
    
    
def convert_jday_to_dm(year,jday):
    '''convertit le jour de l'année en jour et mois de l'année (prend en compte année bissextile)'''
    TOTAL_DAYS = jday
    YEAR = year
    startDate = datetime(year=YEAR, month=1, day=1)
    daysToShift = TOTAL_DAYS - 1
    endDate = startDate + timedelta(days=daysToShift)
    month = endDate.month
    day = endDate.day
    return day,month
    
def days_to_dates(year1,jday1,year2,jday2):
    '''obselete'''
    day1, month1 = convert_jday_to_dm(year1,jday1)
    day2, month2 = convert_jday_to_dm(year2,jday2)
    
    sdate, edate = pd.to_datetime({'year': [year1, year2],
                   'month': [month1, month2],
                   'day': [day1, day2]})
    dates_range = pd.date_range(sdate,edate-timedelta(days=1),freq='d')
    return dates_range
    
    
def compute_dates_range(date1,date2,blend=False):
    sdate, edate = pd.to_datetime({'year': [date1[0], date2[0]],
                   'month': [date1[1],date2[1]],
                   'day': [date1[2], date2[2]]})
    if blend==True:
        dates_range = pd.date_range(sdate,edate-timedelta(days=1),freq='12h')
        if date1[0]!=date2[0]: #on retire la date entre 2001 et 2002 car pas gérée par le code
            dates_range = dates_range.drop('2001-12-31 12:00:00') #on retire cette date car elle nécessite de chevaucher entre 2001 et 2002 pour la calculer
    else:
        dates_range = pd.date_range(sdate,edate-timedelta(days=1),freq='d')
    return dates_range
    
    
def date_to_day(date):
    '''Fonction permettant de convertir les dates en jours de l'année (devra permettre de scinder en deux parties les dates si elles passent à une autre année -> 
    cette focntion permettra alors de faire dune fonction plus générale de plot qui permettra de choisir les dates de début et de fin du plot 
    la date doit être donnée sous forme de dictionnaire avec year,month,day
    ces valeurs sont alors prêtes à être utilisées pour '''
    beg_year = datetime(date[0],1,1)
    true_date = datetime(date[0],date[1],date[2])
    jday = true_date-beg_year
    return jday.days+1


def compute_evolution(date1,date2,veloc_range=None,baz_range=None,plot_blobs=False, clim=None, blend=False):
    '''On doit lui input les dates sous forme de tuple (year,month,day)
    Si l'argument blend est passé en true, alors il va mélanger les beams sucessifs, permettant ainsi de doubler le nombre de points (12h du jour 1 de blend)'''
    day1 = date_to_day(date1) #calcul du jour de l'année
    day2 = date_to_day(date2)
    year1 = date1[0]
    year2 = date2[0]
    if year1!=year2: #if they are not the same year -> then we need to perform a concatenation over the years 
        
        if blend==True:
            day_range1 = np.arange(day1,366-0.5,0.5)
            day_range2 = np.arange(1,day2-0.5,0.5)
        else:
            day_range1 = range(day1,366) #on fait la range de jours dans l'année pour savoir quels fichiers inclure dans l'array
            day_range2 = range(1,day2) 
        
        veloc_matrix1, baz_matrix1, amplitude_matrix1,incert_veloc_matrix1, incert_baz_matrix1 = extract_velo_baz(year1,day_range1,
                                                                                            veloc_range=veloc_range, baz_range=baz_range, plot_blobs=plot_blobs,
                                                                                            clim=clim, blend=blend) #on récupère tous les fichiers dans l'année 1 et on extrait les valeurs + calcule les velo + baz
        veloc_matrix2, baz_matrix2, amplitude_matrix2,incert_veloc_matrix2, incert_baz_matrix2 = extract_velo_baz(year2,day_range2,
                                                                                            veloc_range=veloc_range, baz_range=baz_range, plot_blobs=plot_blobs,
                                                                                            clim=clim, blend=blend)
        veloc_matrix_all = np.concatenate((veloc_matrix1,veloc_matrix2),axis=1) #on fait la concaténation sur les deux années
        baz_matrix_all = np.concatenate((baz_matrix1,baz_matrix2),axis=1) 
        amplitude_matrix_all = np.concatenate((amplitude_matrix1,amplitude_matrix2),axis=1) 
        incert_veloc_matrix_all = np.concatenate((incert_veloc_matrix1,incert_veloc_matrix2),axis=1)
        incert_baz_matrix_all = np.concatenate((incert_baz_matrix1,incert_baz_matrix2),axis=1) 
        #on a maintenant besoin d'avoir la liste des dates recouvrant la période sélectionnée, cette liste sera utulisée pour les plots
        dates_range = compute_dates_range(date1,date2,blend=blend)
    else:
        if blend==True:
            day_range = np.arange(day1,day2-0.5,0.5)
        else:
            day_range = range(day1,day2)
            
        veloc_matrix_all, baz_matrix_all, amplitude_matrix_all,incert_veloc_matrix_all, incert_baz_matrix_all = extract_velo_baz(year1,day_range, 
                                                                                                        veloc_range=veloc_range, baz_range=baz_range, plot_blobs=plot_blobs,
                                                                                                        clim=clim, blend=blend) #peu importe qu'on utilsie l'anéne de la première date ou de la deuxième puisque la même
        dates_range = compute_dates_range(date1,date2,blend=blend)
    return dates_range, veloc_matrix_all, baz_matrix_all, amplitude_matrix_all,incert_veloc_matrix_all, incert_baz_matrix_all    


def days_since_install(date):
    '''Calcule le nombre de jours depuis l'installation, à partir de date au format  datetime; permet alors de vérifier pour une date donnée si il y a de la donnée de suffisemment
    de stations  (il faudra donc utiliser l'availability matrix, mais celle auquel on a retiré les 4 mauvaises stations, parce que il faut par qu'elles soient comptées !
    on lui feed la date déjà en datetime format
    '''
    sdate = datetime(2001,6,24) #date de départ c'est le 24 juin 2006, = le jour 0
    day_passed = date-sdate
    return day_passed.days    


def remove_days(dates_range,veloc_matrix, baz_matrix,amplitude_matrix,incert_veloc_matrix,incert_baz_matrix,threshold_sta):
    '''Cette fonction trouve pour chaque date de dates_range si il y a au moins threshold_sta nombre de stations qui enregistre ce jour là
    si c'est le cas alors le jour est conservé, sinon le jour n'est pas gardé, ce qui fait qu'on aura plus de datapoints venant de jours sans
    ou bien avec trop peu de données'''
    print(f'Initial number of dates : {len(dates_range)}')
    availability_matrix = np.loadtxt('availability_matrix_nobad')
    freq_number = len(veloc_matrix[:,0]) #on a calculé la vitesse et le baz pour certaines fréquences (et leurs fréquences envirnnantes)
    kept_dates = []
    kept_veloc = np.empty((freq_number,0))
    kept_veloc_incert = np.empty((2,0))
    kept_baz = np.empty((freq_number,0))
    kept_baz_incert = np.empty((2,0))
    kept_amplitude = np.empty((freq_number,0))
    for i in range(len(dates_range)):
        if dates_range[i].hour==12: #si on fait un blend donc, car c'est pas un day full!
            idx1 = days_since_install(dates_range[i-1]) #faut retirer 12h, donc on prend juste celui en i-1
            idx2 = days_since_install(dates_range[i+1]) #faut rajouter 12h  et ici on prend celui 12 heure plus tard donc en i+1!
            sta_number1 = np.count_nonzero(availability_matrix[:,idx1]==1) #nombre de stations du jour 1
            sta_number2 = np.count_nonzero(availability_matrix[:,idx1]==1) #nombre de stations du jour 2
            sta_number = np.min([sta_number1,sta_number2])
        else: #si on fait un blend ou pas, mais que on tombe sur un day full, au lieu du blend
            idx = days_since_install(dates_range[i]) #ça correspond alors à l'indice dans l'availability matrix
            sta_number = np.count_nonzero(availability_matrix[:,idx]==1) #on compte le nombre de station enregistrant correctement ce jour là
            
        if sta_number >= threshold_sta: #si pour un jour donc idx donné la somme des stations enregistrant est inférieure au threshold 
            kept_dates.append(dates_range[i])
            kept_veloc = np.concatenate((kept_veloc,veloc_matrix[:,i].reshape((freq_number,1))),axis=1)
            kept_veloc_incert = np.concatenate((kept_veloc_incert,incert_veloc_matrix[:,i].reshape((2,1))),axis=1)
            kept_baz = np.concatenate((kept_baz,baz_matrix[:,i].reshape((freq_number,1))),axis=1)
            kept_baz_incert = np.concatenate((kept_baz_incert,incert_baz_matrix[:,i].reshape((2,1))),axis=1)
            kept_amplitude = np.concatenate((kept_amplitude,amplitude_matrix[:,i].reshape((freq_number,1))),axis=1)
    print(f'Number of dates remaining after applying station threshold: {len(kept_dates)} ({len(dates_range)-len(kept_dates)} removed)')
    return kept_dates, kept_veloc, kept_baz, kept_amplitude,kept_veloc_incert,kept_baz_incert


def remove_nowave(dates_range, veloc_matrix, baz_matrix, amplitude_matrix,incert_veloc_matrix,incert_baz_matrix):
    '''Remove nodat est supporté que pour le mode advanced '''
    print(f"Initial number of dates before rm_nowave: {len(dates_range)}")
    freq_number = len(veloc_matrix[:,0])
    kept_dates = []
    kept_veloc = np.empty((freq_number,0))
    kept_veloc_incert = np.empty((2,0))
    kept_baz = np.empty((freq_number,0))
    kept_baz_incert = np.empty((2,0))
    kept_amplitude = np.empty((freq_number,0))
    for i in range(len(dates_range)):
        if veloc_matrix[0,i] != 999: #vitesse set à 999 si il trouve rien!!
            kept_dates.append(dates_range[i])
            kept_veloc = np.concatenate((kept_veloc,veloc_matrix[:,i].reshape((freq_number,1))), axis=1)
            kept_veloc_incert = np.concatenate((kept_veloc_incert,incert_veloc_matrix[:,i].reshape((2,1))),axis=1)
            kept_baz = np.concatenate((kept_baz,baz_matrix[:,i].reshape((freq_number,1))), axis=1)
            kept_baz_incert = np.concatenate((kept_baz_incert,incert_baz_matrix[:,i].reshape((2,1))),axis=1)
            kept_amplitude = np.concatenate((kept_amplitude,amplitude_matrix[:,i].reshape((freq_number,1))), axis=1)
    print(f"Number of dates after rm_nowave : {len(kept_dates)} ({len(dates_range)-len(kept_dates)} removed)")
    return kept_dates, kept_veloc, kept_baz, kept_amplitude,kept_veloc_incert, kept_baz_incert


def remove_badbeams(dates_range,veloc_matrix, baz_matrix,amplitude_matrix,incert_veloc_matrix,incert_baz_matrix, dates_to_rm=None):
    '''Cette fonction permet de retirer les data points associés aux beams qui ont une mauvaise allure, il prend en argument une liste contenant toutes les dates à retirer 
    lui permettant alors de trouver dans dates range les indices des matrices dne devant pas être gardés'''
    print(f'Removing listed bad beams... {len(dates_to_rm)} beams to remove in total')
    freq_number = len(veloc_matrix[:,0]) #on a calculé la vitesse et le baz pour certaines fréquences (et leurs fréquences envirnnantes)
    kept_dates = []
    kept_veloc = np.empty((freq_number,0))
    kept_veloc_incert = np.empty((2,0))
    kept_baz = np.empty((freq_number,0))
    kept_baz_incert = np.empty((2,0))
    kept_amplitude = np.empty((freq_number,0))
    for i in range(len(dates_range)):
        counter = 0
        for j in range(len(dates_to_rm)):
            counter+=np.count_nonzero(dates_to_rm[j]==dates_range[i]) #obligé d'itérer comme ça avec ces objets 
        if counter==0: #si counter = 0 ça veut dire que aucun des jours à remove correspond au jour considéré, donc que le jour doit bien être cobservé !  
            kept_dates.append(dates_range[i])
            kept_veloc = np.concatenate((kept_veloc,veloc_matrix[:,i].reshape((freq_number,1))),axis=1)
            kept_veloc_incert = np.concatenate((kept_veloc_incert,incert_veloc_matrix[:,i].reshape((2,1))),axis=1)
            kept_baz = np.concatenate((kept_baz,baz_matrix[:,i].reshape((freq_number,1))),axis=1)
            kept_baz_incert = np.concatenate((kept_baz_incert,incert_baz_matrix[:,i].reshape((2,1))),axis=1)
            kept_amplitude = np.concatenate((kept_amplitude,amplitude_matrix[:,i].reshape((freq_number,1))),axis=1)
    print(f'Number of dates remaining : {len(kept_dates)} ({len(dates_range)-len(kept_dates)} removed)')
    return kept_dates, kept_veloc, kept_baz, kept_amplitude,kept_veloc_incert, kept_baz_incert


def remove_low_amp(dates_range, veloc_matrix, baz_matrix, amplitude_matrix,incert_veloc_matrix,incert_baz_matrix, amp_threshold):
    '''Cette fonction sert à retirer tous les data points où l'amplitude est inférieure à un threshold donné
    le threshold qu'on utilise est directement le threshold en amplitude '''
    print(f"Initial number of dates before applying amplitude threshold : {len(dates_range)}")
    freq_number = len(veloc_matrix[:,0])
    kept_dates = []
    kept_veloc = np.empty((freq_number,0))
    kept_veloc_incert = np.empty((2,0))
    kept_baz = np.empty((freq_number,0))
    kept_baz_incert = np.empty((2,0))
    kept_amplitude = np.empty((freq_number,0))
    for i in range(len(dates_range)):
        if amplitude_matrix[0,i] >= amp_threshold: 
            kept_dates.append(dates_range[i])
            kept_veloc = np.concatenate((kept_veloc,veloc_matrix[:,i].reshape((freq_number,1))), axis=1)
            kept_veloc_incert = np.concatenate((kept_veloc_incert,incert_veloc_matrix[:,i].reshape((2,1))),axis=1)
            kept_baz = np.concatenate((kept_baz,baz_matrix[:,i].reshape((freq_number,1))), axis=1)
            kept_baz_incert = np.concatenate((kept_baz_incert,incert_baz_matrix[:,i].reshape((2,1))),axis=1)
            kept_amplitude = np.concatenate((kept_amplitude,amplitude_matrix[:,i].reshape((freq_number,1))), axis=1)
    print(f"Number of dates after amp_threshold : {len(kept_dates)} ({len(dates_range)-len(kept_dates)} removed)")
    return kept_dates, kept_veloc, kept_baz, kept_amplitude,kept_veloc_incert, kept_baz_incert


def plot_evolution(dates_range,veloc_matrix,baz_matrix,amplitude_matrix,incert_veloc_matrix,incert_baz_matrix,freqs_to_plot=pl_updated['fn'], threshold_sta=0,
                   rm_nowave=False, figname=None, show_amplitude=False, rm_badbeams=None, amp_threshold=None, show_baz_incert=False,
                   show_veloc_incert=False):
    '''cette fonction permet  sélectionner les fréquences utilisées pour l'interprétation
    L'option  threshold_sta permet de retirer les data points où moins de stations qu'annoncé par le threshold
    sont en train d'enregistrer (se base sur le décompte du nombre de 1 dans l'availability matrix -> on utilise 
    la version updaté avec -1 pour les jours avec des )
    
    Si on utilise le mode advanced alors aucune freq particulière à plot -> on utilise alors None
    
    On lui rajoute aussi la fonction rm_nodat qui retire les points où le clustering nous a pas permis de trouver de cluster correspondant à l'onde
    d'intére^t  (retire les points où la vitesse = 0'''
    plt.close('all')
    
    #first we remove what we don't want to see in our data
    if threshold_sta >0:
        dates_range, veloc_matrix, baz_matrix, amplitude_matrix, incert_veloc_matrix,incert_baz_matrix = remove_days(dates_range, veloc_matrix, baz_matrix,amplitude_matrix,
                                                                                    incert_veloc_matrix,incert_baz_matrix,threshold_sta)
    #on va chercher la freq range en utilisant un fichier quelconque
    beam_file = f"{in_['beam_dir']}/beam_{compo_to_plot}_2001_day_275.h5"
    data = h5py.File(beam_file,'r')
    freqs = data['ff']['sfreq'][()]
    ifreq = m_beam.make_short_freq(pl_updated['fn'][0],freqs,fw=pl_updated['fw'],nf=0) #calcule la gamme de fréquences utilisées avec le compute evolution!
    #we can plot now
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(13,5))
    # fig.suptitle(f"Temporal evolution of Velocity and Back Azimuth for frequencies in range {round(freqs[ifreq[0]],2)}-{round(freqs[ifreq[-1]],2)}Hz")
    ax1.set_ylabel('Velocity (km/s)', color='#1f77b4')
    ax2.set_ylabel('Back Azimuth (°)', color='#1f77b4')
    # ax2.set_xlabel('Dates')
    
    for freq in freqs_to_plot:
        idx_freq = pl_updated['fn'].index(freq)  #pour la fréquence qu'on veut plotter, il faut trouver dans le pl_updated['fn'] quel est son indice afin de plotter la bonne composante!
            
        if rm_nowave ==True: #retire les jours où le clustering n'a pas trouvé de cluster dans la range donnée -> FONCTIONNE QUE POUR 1 FRÉQUENCE
            dates_range, veloc_matrix, baz_matrix,amplitude_matrix,incert_veloc_matrix,incert_baz_matrix = remove_nowave(dates_range,veloc_matrix, baz_matrix,
                                                                                                                        amplitude_matrix,incert_veloc_matrix, incert_baz_matrix)
        if amp_threshold!=None:
            dates_range, veloc_matrix, baz_matrix,amplitude_matrix, incert_veloc_matrix, incert_baz_matrix = remove_low_amp(dates_range,veloc_matrix, baz_matrix,
                                                                                                                amplitude_matrix,incert_veloc_matrix,incert_baz_matrix, amp_threshold)    
        if rm_badbeams!=None: #on retire la liste des dates où les beams sont mauvais
            dates_range, veloc_matrix, baz_matrix,amplitude_matrix,incert_veloc_matrix,incert_baz_matrix = remove_badbeams(dates_range,veloc_matrix, baz_matrix,
                                                                                                                        amplitude_matrix,incert_veloc_matrix,incert_baz_matrix, rm_badbeams)
        # NOW WE CAN ACTUALLY START TO PLOT STUFF 
        
        if show_veloc_incert==True:
            markers_veloc,caps, bars_veloc = ax1.errorbar(dates_range,veloc_matrix[idx_freq,:],yerr=incert_veloc_matrix, marker='.', ls='none')
            [bar.set_alpha(0.3) for bar in bars_veloc]
        else:
            ax1.plot(dates_range,veloc_matrix[idx_freq,:],label=f"{freq} Hz", marker='.', linewidth=0)
            
        if show_baz_incert==True: #on peut plotter le baz avec ou sans error bar WOAW 
            markers_baz, caps, bars_baz = ax2.errorbar(dates_range, baz_matrix[idx_freq,:], yerr=incert_baz_matrix, marker='.', ls='none')
            [bar.set_alpha(0.3) for bar in bars_baz]
        else:
            ax2.plot(dates_range,baz_matrix[idx_freq,:],label=f"{freq} Hz", marker='.', linewidth=0)
        
        #enveloppe comme dans leur papier 
        # ax1.plot(dates_range,np.ones(len(dates_range))/0.21, linewidth=1, color='green')
        # ax1.plot(dates_range,np.ones(len(dates_range))/0.19, linewidth=1, color='green')
        
        ax1.set_ylim(3.5,6)
        ax2.set_ylim(0,360)    
            
        if show_amplitude==True:
            ax1_bis = ax1.twinx()
            ax2_bis  = ax2.twinx()
            ax1_bis.plot(dates_range,amplitude_matrix[idx_freq,:], marker='.', color='red')
            ax2_bis.plot(dates_range,amplitude_matrix[idx_freq,:], marker='.', color='red')
            if pl_updated['db']==True:
                ax1_bis.set_ylabel('Amplitude (dB)', color='red')
                ax2_bis.set_ylabel('Amplitude (dB)', color='red')
                ax1_bis.set_ylim(-69,-59)
                ax2_bis.set_ylim(-69,-59)
            else:
                ax1_bis.set_ylabel('Amplitude', color='red')
                ax2_bis.set_ylabel('Amplitude', color='red')
                ax1_bis.set_ylim(0.0011,0.003)
                ax2_bis.set_ylim(0.0011,0.003)
            
        if figname!=None:
            print('file saved')
            plt.savefig(figname,bbox_inches='tight',dpi=300)        
        
        return dates_range, veloc_matrix[0,:], baz_matrix[0,:], amplitude_matrix[0,:]
       
    
################################ POUR LA MÉTHODE NORMALE DE EXTRAC V BAZ !!! ET POUR POUVOIR PLOT LA POSITION 

def find_local_max(Z,U,veloc_range=None,baz_range=None,plot_blobs=False, clim=None, thresh_amp=0.95):
    '''Cette fonction a pour vocation de trouver les coordonées du maximum local dans une gamme de vitesse et baz donnée  à l'aide de clustering
    Elle est employée par le mode normal dans la fonction extract velo_baz 
    Fonction séparée pour pouvoir initialement être ajoutée aux plots des panneaux de beamforming pour vérifier que le maximum est correctement localisé!
    La location en output est modifiée par rapport à la location en sortie de np where, de telle sorte à ce que le premier indice = x et deuxième indice = y
    cette fnction retrouve par ailleurs le maximum d'amplitude du cluster, cettte amplitude sera alors plottée dans la timeserie'''
    du   = U[1]-U[0]
    y, x = np.mgrid[min(U)-du/2:max(U)+du/2:U.size+1+1j,min(U)-du/2:max(U)+du/2:U.size+1+1j]
    #On trouve la location des peaks, on cherchera par la suite lequels de ces peaks sont présents dans les blobs!!
    fp = findpeaks(method='topology', verbose=0)
    peaks = fp.fit(Z)
    #on convertit Z en uint8 pour l'utiliser ave cle bloc detector 
    coeff = 255/(np.max(Z)-np.min(Z))
    Z_gray = np.uint8(Z*coeff-np.min(Z)*coeff)
    blobs = blob_log(Z_gray) #c'est bon on a nos blobs,
    #on devrait maintenant avoir extrait les clusters, il faut à présent vérifier pour chaque cluster s'il respecte les conditions données et si c'est le cas alors
    #il ses coordonnées sont gardées 
    velocity = 999 # fonctionnera avec l'option nowave -> on retirera le data point s'il n'avait pas trouvé de cluster respectant les critères!
    velocity_incert = (999,999)
    backazimuth = 999
    backazimuth_incert = (999,999)
    amplitude = -999999999999999999999 
    true_loc = ([0],[0]) #il return aussi la loc pour pouvoir permettre de plotter par la suite le cluster trouvé
    
    if plot_blobs==True:
        plt.close('all')
        fig, ax = plt.subplots(1)
        extent = np.min(x[0,:]), np.max(x[0,:]), np.min(y[:,0]), np.max(y[:,0]) #sert juste pour le plot en imshow, ne pas utilsier x et y pour calculer des valeurs !!
        if clim!=None: 
            vmin, vmax = clim[0], clim[1]
        else:
            vmin = Z.min()
            vmax = Z.max()
        im = ax.imshow(Z,vmin=vmin, vmax=vmax,origin='lower',extent=extent,cmap=pl_updated['clmap'])
        fig.colorbar(im)
        ax.set_ylabel('slowness (s/km)')
        ax.set_xlabel('slowness (s/km)')
        
        #on plot les différents pics extraits avec axtract peaks, permet de vérifier qu'il fonctionne bien
        peak_number_max = np.count_nonzero(peaks['Xranked']>0)
        for peak_num in range(1,peak_number_max):
            loc_plot = np.where(peaks['Xranked']==peak_num)
            # ax.scatter(U[loc_plot[1]],U[loc_plot[0]], c='green') #commenter pour pas plot les peaks 
        
        for blob in blobs:#on plot les divers blobs
            y_blob, x_blob, r_blob = blob
            y_blob, x_blob = int(y_blob), int(x_blob)
            x_blob_plot,y_blob_plot = U[x_blob],U[y_blob]
            du = x[0,1]-x[0,0] #avoir le pas spatial
            r_blob_plot = r_blob*du #convertit la distance de pixels à vraie distance !
            c_blob = plt.Circle((x_blob_plot,y_blob_plot),r_blob_plot, color='green', linewidth=1, fill=False, clip_on=True)
            ax.add_patch(c_blob)
        
        ax.set_xlim(extent[0],extent[1])
        ax.set_ylim(extent[2],extent[3])
    
    for blob in blobs: 
        y_blob, x_blob, r_blob = blob #seerivra à trouver les incertitudes ainsi qu'à s'assurer que le pic trouvé est
        y_blob, x_blob = int(y_blob), int(x_blob)
        #On doit trouver le peak avec findpeaks qui est dans le blob et a la plus grande amplitude possible!
        peak_number = 0
        conv = False #on va itérer à travers les différents pics et en trouver un dans le blob
        peak_number_max = np.count_nonzero(peaks['Xranked']>0) #compte le nombre de pics positifs, comme ça si on dépasse le nombre de peaks, alors on arrête de chercher 
        while conv==False and peak_number<peak_number_max: 
            peak_number+=1
            loc = np.where(peaks['Xranked']==peak_number)
            #ddoit maintenant faire une condition pour que le peak dans le blob
            if np.sqrt((loc[0][0]-y_blob)**2+(loc[1][0]-x_blob)**2)<r_blob:
                conv=True
        
        peak_veloc = compute_velo(loc, U)
        peak_baz = compute_baz(loc, U)
        peak_amplitude = Z[loc[0][0],loc[1][0]]
        
        
        incert_veloc_inf = 99999999999
        incert_veloc_sup = 0
        incert_baz_inf = 99999999999
        incert_baz_sup = 0
        
        for j in range(len(U)): #pour tous les points du cluster
            for i in range(len(U)): #l'maplitude threshold se base sur l'amplitude du peak principale du blob considéré
                if np.sqrt((j-y_blob)**2+(i-x_blob)**2)<r_blob and Z[j,i]>thresh_amp*peak_amplitude: #si le pixel considéré est dans le blob, alors on va calculer la vitesse et le BAZ du pixel
                    temp_loc = ([j],[i]) #les coordonnées utilisées sont direct les indexs 
                    temp_veloc  = compute_velo(temp_loc, U) #on calcule la vitesse du point associé 
                    temp_baz  = compute_baz(temp_loc, U) #on calcule alors le baz associé au point 
                    
                    if plot_blobs==True:
                        ax.scatter(U[i],U[j],s=1,color='deeppink') #juste pour montrer quels points sont considérés quand on calcule l'incertitude !!!
                    
                    if temp_veloc < incert_veloc_inf:
                        incert_veloc_inf = temp_veloc
                    if temp_veloc > incert_veloc_sup: #same s'il est plus grand
                        incert_veloc_sup = temp_veloc
                    if temp_baz < incert_baz_inf: #si le point a un baz inférieur au plus faible baz actuel du cluster alors il le remplace
                        incert_baz_inf = temp_baz
                    if temp_baz > incert_baz_sup: #same s'il est plus grand
                        incert_baz_sup = temp_baz
                
        if  veloc_range[0]<= peak_veloc <= veloc_range[1] and baz_range[0] <= peak_baz <= baz_range[1] and peak_amplitude>amplitude:
            #si amplitude d'un autre blob est supérieure à celle du précédent, alros il sera remplacé 
            velocity = peak_veloc
            velocity_incert = (velocity-incert_veloc_inf, incert_veloc_sup-velocity)
            backazimuth = peak_baz
            backazimuth_incert = (backazimuth-incert_baz_inf, incert_baz_sup-backazimuth) #on fait un tuple qui contient le min et max baz du cluster ->  donne alors une sorte d'incertitude
            amplitude = peak_amplitude  #c'est bon parce que loc contient d'abord indice den y
            true_loc = (loc[1],loc[0]) #on réarrange pour que ce soit les x en premeir puis les y, sert à plotter sur la figure
    
    if plot_blobs==True and true_loc!=([0],[0]): #rajoute le 0,0 car c'est ce qu'on met quand on sait pas où est le max, du coup là on le plot plus 
        ax.scatter(U[true_loc[0]], U[true_loc[1]], marker='*', s=80, edgecolors='black', color='pink')
        
    return true_loc, velocity, backazimuth, amplitude, velocity_incert ,backazimuth_incert 
    
#rolling averages pour essayer d'avoir quelque chose de stable dans le temps 

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_clean(dates_range, veloc, baz, amplitude, figname=None):
    '''Pour plot la velocity, le baz et l'maplitude à partir des vecteurs déjà extraits de la matrice et qui ont pu ou pas être modifiés (ma, etc)'''
    #we can plot now
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(13,5))
    fig.suptitle('Temporal evolution of Velocity and Back Azimuth')
    ax1.set_ylabel('Velocity (km/s)', color='#1f77b4')
    ax2.set_ylabel('Back Azimuth (°)', color='#1f77b4')
    ax2.set_xlabel('Dates')

    ax1.plot(dates_range,veloc)
    ax2.plot(dates_range,baz)
    ax2.set_ylim(0,360)
    
    ax1_bis = ax1.twinx()
    ax2_bis  = ax2.twinx()
    ax1_bis.plot(dates_range,amplitude, color='red')
    ax2_bis.plot(dates_range,amplitude, color='red')
    
    if pl_updated['db']==True:
        ax1_bis.set_ylabel('Amplitude (dB)', color='red')
        ax2_bis.set_ylabel('Amplitude (dB)', color='red')
        ax1_bis.set_ylim(-69,-59)
        ax2_bis.set_ylim(-69,-59)
    else:
        ax1_bis.set_ylabel('Amplitude', color='red')
        ax2_bis.set_ylabel('Amplitude', color='red')
        ax1_bis.set_ylim(0,0.01)
        ax2_bis.set_ylim(0,0.01)
    
    if figname!=None:
        plt.savefig(figname,bbox_inches='tight', dpi=300)
        
        

#################################################################################################################################################################

#FONCTIONS DE BACKPROJECTION 


######################################################################################################


def make_date_range_backpropag(date1, date2, step='12h', badbeam_list=None):
    availability_matrix = np.loadtxt('availability_matrix_nobad')
    sdate, edate= pd.to_datetime({'year': [date1[0], date2[0]],
                   'month': [date1[1],date2[1]],
                   'day': [date1[2], date2[2]]})
    dates_range = pd.date_range(sdate,edate-timedelta(days=1),freq=step)
    if date1[0]!=date2[0]: #on retire la date entre 2001 et 2002 car pas gérée par le code
            try:
                dates_range = dates_range.drop('2001-12-31 12:00:00')
            except:
                print('Date was not in range')
    a = len(dates_range)
    print(f'Date range created with {len(dates_range)} dates')
    #doit utiliser fonction en installed days pour trouver les indices dansavailability matrix
    station_threshold = 20
    for date in dates_range:
        if date.hour==0: #si c'est en heure pleine seulement qu'on va chercher à vérifier si on a de la donnée du jour même 
            idx_day = (date-datetime(2001,6,24)).days #calcule le nombre de jour depuis le 24 juin qui est le jour d'installation 0 = le Jday 175
            station_number = np.count_nonzero(availability_matrix[:,idx_day]==1)
            if (date-datetime(2002,1,1)).days+1==102: #calcule le jday pour savoir si c'est le j 102 qui bug
                station_number = 0 #on met alors à 0 car bugged
        elif date.hour==12:
            idx_day1 = (date-datetime(2001,6,24)).days
            idx_day2 = idx_day1+1
            station_number1 = np.count_nonzero(availability_matrix[:,idx_day1]==1)
            station_number2 = np.count_nonzero(availability_matrix[:,idx_day2]==1)
            station_number = np.min([station_number1,station_number2])
            if (date-datetime(2002,1,1)).days+1==102 or (date-datetime(2002,1,1)).days+1==101: #pareil ici, si la date est 4 avril 12h (102+103) ou 3 avril 12h (101+102)
                station_number=0 #alors ça dégage!
        if station_number<station_threshold: #faut avoir au moins 20 bonnes stations pour que beam pas trop mal
            dates_range = dates_range.drop(date) #dans ce cas on drop la date!
    print(f'{len(dates_range)} dates left after removing dates with less than {station_threshold} stations ({a-len(dates_range)} removed)')
    a = len(dates_range)
    #qui plus est on supprime aussi une liste de beams 
    if badbeam_list!=None:
        for badbeam in badbeam_list:
            try:
                dates_range = dates_range.drop(badbeam)
            except:
                if  ((badbeam-dates_range[0]).days+1)>0  and ((dates_range[-1]-badbeam).days+1)>0: #si la date de fin est plus tard que date du beam, alors c'est que date a déjà été retirée! 
                    print('Date already gone!')
    print(f'{len(dates_range)} Dates left after removing those with bad beams ({a-len(dates_range)} removed)')
    return dates_range


def extract_Z_backpropagation(date):
    year = date.year
    jday = (date-datetime(year,1,1)).days+1   #calcul le jday de la date
    if date.hour==12: #cas où il faut blend les deux beams 
        day_first = jday #FOCNTION QUI CONVERTIT  LA DATE EN JOUR JULIEN 
        if day_first!=365:
            day_second = day_first+1
        else:
            day_second = 1 #prend en compte l'effet du switch entre 2001 et 2002!
        
        data1 = h5py.File(f"{in_['beam_dir']}/beam_{bb['compo']}_{year}_day_{day_first:03}.h5",'r') 
        data2 = h5py.File(f"{in_['beam_dir']}/beam_{bb['compo']}_{year}_day_{day_second:03}.h5",'r')
        fk1   = data1['fk'][()]
        fk2   = data2['fk'][()]
        freqs = data1['ff']['sfreq'][()]   #ça devrait pas matter si on prend depuis le 1 ou le 2
        U    = data1['bb']['U'][()] * 10**3 #
    else: 
        day = jday  
        data = h5py.File(f"{in_['beam_dir']}/beam_{bb['compo']}_{year}_day_{day:03}.h5",'r') #il importe le fichier 
        fk   = data['fk'][()]
        freqs = data['ff']['sfreq'][()]
        U    = data['bb']['U'][()] * 10**3
    
    #c'est bon on a tout maintenant on a plus qu'à importer Z!! 
    for j in range(len(pl_updated['fn'])):
        ifreq = m_beam.make_short_freq(pl_updated['fn'][j],freqs,fw=pl_updated['fw'],nf=0)
        if date.hour==12:   #pareil, si on est sur du 12h alors faut que ça chevauche les deux jours  
            Z1 = np.mean(np.abs(fk1[ifreq,:,:]),axis=0).transpose()
            Z2 = np.mean(np.abs(fk2[ifreq,:,:]),axis=0).transpose()
            Z  = (Z1+Z2)/2 #on fait la moyenne des beams des deux jours différents !
        else: #si int(days) = days alors c'est qu'on est sur un jour spécifique et donc on a déjà son Z grâce au fk
            Z = np.mean(np.abs(fk[ifreq,:,:]),axis=0).transpose() #on a supprimé l'option pour faire
    return Z #pas beosoin de ressortir le 0.8 à 1.3 Hz 


def find_p(distance, p_option='ray',model=None,Phase=None,source_depth=None):
    '''fonction servant à trouver la slowness apparente, basée sur la distance entre le point et le barycentre. 2 modes dispos:
    ray pour raytracing et fw pour full waveform 
    on lui feed direct le vecteur dis et slow au lieu de lire le fichier à chaque fois si mode fw'''
    if p_option=='ray':
        distances = np.linspace(distance-1,distance+1,3)*cake.m2d #converison de la distance en degrés 
        arrivals = model.arrivals(distances, phases=Phase, zstart=source_depth)
        best_efficiency = 0
        #il y a probablement plusieurs arrivals à la même distance, on choisi alors cette qui a la plus forte efficiency 
        p=1.0
        for arrival in arrivals:
            if arrival.efficiency()>=best_efficiency:
                best_efficiency = arrival.efficiency()
                p = arrival.p*np.pi/180/cake.d2m*1000. #on récupère alors le paramètre de rai associé à cette meilleure arrivée!
    else:
        p = model[1,[np.argmin(abs(model[0,:]-distance))]][0] #là dans le slowness profile on récupère la valeur la plus adaptée au vue de la distance : il faudrait 
                                                                #ajouter une donction de distance en mode si distance trop élevée alors 
    return p
    
def compute_backpropag(sdate, edate, step,latitudes, longitudes, badbeam=None,v_model='california_model_v3.m',station_file='stations_rm_nobad.txt',
                       s_depth=0, plot_map=False, p_option='ray'):
    
    print(f"Using beams from {in_['beam_dir']} with {bb['compo']} component")
    dates_range = make_date_range_backpropag(sdate,edate,step,badbeam) # il trouve les diverses dates à plot 
    # on calcule dès maintenant les valeurs de baz et de slowness associées aux divers points du panneau, et on créé alors 2 matrices dans lesuqelles ces valeurs seront  
    U = bb['U']* 1000 #direct importé depuis variables.py
    slow_mesh, baz_mesh = np.zeros((len(U),len(U))), np.zeros((len(U),len(U)))   
    for j in range(len(U)):
        for i in range(len(U)):
            slow_mesh[j,i] = 1/compute_velo(([j],[i]),U) 
            baz_mesh[j,i] = compute_baz(([j],[i]),U) 
    ################################################################### 
    #on calcule aussi les coordonnées du barycentre pour trouver le backazimuth et la distance du point gps vis à vis du barycentre, besoin d'adapter 
    #le fichier servant à calculer le barycentre en fonction des circonstances, car pas tjr même nb de stations !!    
    lon_bary, lat_bary,height_bary   = compute_barycenter(station_file) #calcule coordonnées du barycentre
    ############################################################## définition de constantes pour la backprogation
    if p_option =='ray': 
        model = cake.load_model(v_model)
        source_depth = s_depth * 1000. #conversion en m
        Phase = cake.PhaseDef('P')
    else:
        model = np.loadtxt(v_model) #on utilise le même argument que 
        source_depth, Phase = None, None
    ###################################################
    
    Z_mesh = np.zeros((len(latitudes),len(longitudes),len(dates_range)))
    
    print('Computing backpropagations, please wait...')
    for d, date in enumerate(tqdm(dates_range)):
        #on extract le beam du jour!
        Z = extract_Z_backpropagation(date)
        for j, lat in enumerate(latitudes):
            for i, lon, in enumerate(longitudes):
                #on calcule la distance en km entre le bary et le pt de coords 
                distance, baz, az = gps2dist_azimuth(lat_bary,lon_bary,lat,lon) #va de array vers source dans ce sens, donc first est le baz qu'on a 
                if distance>=np.min(model[0,:]) and distance<=np.max(model[0,:]):   #############" condition pour que si point en dehors de la range alors direct on met nan dans la matrice !! 
                    p = find_p(distance=distance,p_option=p_option,model=model, source_depth=source_depth, Phase=Phase)
                    #maintenant que l'on a le paramètre de rai, et le baz, on va pouvoir trouver la position de ce best dans les mesh !!
                    #puisque la 
                    if abs(p)>0.4242: #s'il est en dehors du cadre
                        Z_mesh[j,i,d] = 0   
                    else: #maintenant on va réellement devoir chercher les coordonnées du point ainsi que la valeur de Z associée ! 
                        conv = False
                        ecart = 0.002
                        while conv==False: #on lui fait un faible écart au départ pour essayer d'avoir une bonne résolution, mais si ça empêche d'avoir le cercle alors increased
                            ecart = 2*ecart
                            slow_y_low, slow_x_low  = np.where(slow_mesh<p-ecart) 
                            slow_y_high, slow_x_high = np.where(slow_mesh>p+ecart) #on cherche à créer un cercle de 1 sur lequel on cherche l'index du bon BAZ
                            circle = np.ones(slow_mesh.shape)
                            circle[slow_y_low, slow_x_low] = 0
                            circle[slow_y_high, slow_x_high] = 0
                            if np.sum(np.sum(circle))>0:
                                conv = True
                        #c'est bon on a notre cercle !
                        baz_correct = -999
                        loc_correct = -999 #on donne des valeurs fausses juste pour initiliser
                        y_circle, x_circle = np.where(circle==1) #on a une liste de x et de y du circle, on va alors chercher pour ces coordonnées dans le mesh où c'est le plus proche du baz 
                        idef = np.argmin(abs(baz_mesh[y_circle, x_circle]-baz)) #on cherche indice dans liste de points du cercle qui permet d'avoir le baz le plus proche 
                        loc_correct = [[y_circle[idef]],[x_circle[idef]]]
                        #on connait maintenant la bonne location en indexs dans Z, il nous suffit de récupérer la valeur dans Z aka le beam! 
                        Z_mesh[j,i,d] = Z[loc_correct[0],loc_correct[1]] #c'est bon ! on a maintenant remplila cellule du mesh avec l'ampltude de Z associée!                            
                else:
                    Z_mesh[j,i,d] = np.nan #si au dela de la la range de dustances du modèle !!!
                
        if plot_map==True:#option pour directement plot quand ça tourne 
            fig = plot_backpropagation(dates_range[d],latitudes,longitudes, Z_mesh[:,:,d])
            n = 0
            for lat in latitudes:
                for lon in longitudes:
                    n+=1
                    plt.text(lon, lat, str(n),color='red',transform=ccrs.PlateCarree())
            
    return dates_range, Z_mesh  
                    


def plot_backpropagation(dates_range, latitudes, longitudes, Z_mesh,folder='test',lims=None, plot_eq=None, station_file='stations_rm_nodbad.txt'):
    stamen_terrain = Stamen(desired_tile_form="RGB", style="terrain-background") 
    
    longitudes_station,latitudes_station,heights_station = extract_coordinates(station_file) #récupère les coordoonées des stations pour le plot
    lon_bary, lat_bary,height_bary   = compute_barycenter(station_file) #calcule coordonnées du barycentre
    
    extent = [longitudes[0], longitudes[-1], latitudes[0], latitudes[-1]]
    
    path = f'/summer/faultscan/user/parisnic/m1_internship/parkfield/BACKPROPAGATION_FIGS/{folder}'
    
    try:
        os.makedirs(path) 
    except:
        if isinstance(dates_range, pd.DatetimeIndex)==True: #pour qu'il print pas si seulement 1 date !
            print('Folder already exists, files will be replaced')
    
    
    if isinstance(dates_range, pd.DatetimeIndex)==True:
        print('Plotting backpropagation at every date ...')
    
    try: #fait avec try comme ça si ça marche pas car 1 seule date, on peut aussi gérer 
        for i in tqdm(range(len(dates_range))):
            plt.close('all')
            plt.figure(figsize=(12, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent(extent)
            ax.add_image(stamen_terrain, 8)
            ax.coastlines()        

            if lims==None:
                non_nan_loc = np.where(np.isnan(Z_mesh[:,:,i])==False)
                lims = [Z_mesh[non_nan_loc,i].min(), Z_mesh[non_nan_loc,i].max()]
              
            hmap = ax.pcolormesh(longitudes, latitudes, Z_mesh[:,:,i],
                 transform=ccrs.PlateCarree(), cmap='jet',alpha=0.4, vmin=lims[0], vmax=lims[1])

            axins = inset_axes(ax,
            width="3%",  # width: 5% of parent_bbox width
            height="100%",  # height: 50%
            loc="lower left",
            bbox_to_anchor=(1.01, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )    
            
            cbar = plt.colorbar(hmap,location='right', cax=axins)
            
            ax.scatter(longitudes_station, latitudes_station, color='forestgreen',edgecolors='black', marker='^', transform=ccrs.PlateCarree())
            # ax.scatter(lon_bary, lat_bary, color='green', marker='*', transform=ccrs.PlateCarree())
            
            if plot_eq!=None: #peut aussi plot la position de l'eq si nécessaire !
                ax.scatter(plot_eq[1],plot_eq[0],c='red', marker='*',s=100) #on échait format lat lon, donc inversion pour le scatter
            
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0, color='gray', alpha=0.5, linestyle='--')

            gl.bottom_labels = False
            gl.right_labels = False

            plt.text(0.4,-0.08,f'{dates_range[i]}',transform=ax.transAxes)

            plt.savefig(f'{path}/map_pcolormesh_{dates_range[i]}.png', bbox_inches='tight',dpi=300)
            
    except:
        plt.close('all')
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent)
        stamen_terrain = Stamen(desired_tile_form="RGB", style="terrain-background") 
        ax.add_image(stamen_terrain, 8)
        ax.coastlines()        
        
        #creating the levels for the cbar :
        if lims==None:
            non_nan_loc = np.where(np.isnan(Z_mesh[:,:])==False)
            lims = [Z_mesh[non_nan_loc].min(), Z_mesh[non_nan_loc].max()]
        
        hmap = ax.pcolormesh(longitudes, latitudes, Z_mesh[:,:],
                 transform=ccrs.PlateCarree(), cmap='jet',alpha=0.4, vmin=lims[0],vmax=lims[1])
        
        axins = inset_axes(ax,
        width="3%",  # width: 5% of parent_bbox width
        height="100%",  # height: 50%
        loc="lower left",
        bbox_to_anchor=(1.01, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
        )
        
        cbar = plt.colorbar(hmap,location='right', cax=axins)
        
        ax.scatter(longitudes_station, latitudes_station, color='red', marker='^', transform=ccrs.PlateCarree())
        ax.scatter(lon_bary, lat_bary, color='green', marker='*', transform=ccrs.PlateCarree())
        
        if plot_eq!=None: #peut aussi plot la position de l'eq si nécessaire !
                ax.scatter(plot_eq[1],plot_eq[0],c='black', marker='*',s=80) #on échait format lat lon, donc inversion pour le scatter
        
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0, color='gray', alpha=0.5, linestyle='--')
        gl.bottom_labels = False
        gl.right_labels = False
        plt.text(0.4,-0.08,f'{dates_range}',transform=ax.transAxes)
        plt.savefig(f'{path}/map_pcolormesh_{dates_range}.png', bbox_inches='tight',dpi=300)
            
            
##################################################################################################################################


#FOCNTIONS DE  GRID SEARCH SLANT STACK POUR LE NOISE 



#######################################################################################################"

def compute_dist2source_idxshift_matrix(longitudes,latitudes,lon_station,lat_station,dist_profile,slowness_profile,fs):
    '''Cette focntion calcule pour une grid fournie la distance des chacun des points de la grille vs chacun des receivers!
    On en profite aussi pour calculer le timeshift à effectuer'''
    dist2source_matrix = np.zeros((len(latitudes),len(longitudes),len(lon_station)))
    idx2shift_matrix = np.zeros((len(latitudes),len(longitudes),len(lon_station)))
    for i, lat in enumerate(latitudes): #loop over the different stations and the different points of the grid to know their relative distances
        for j, lon in enumerate(longitudes):
            for k in range(len(lon_station)):
                dist2source_matrix[i,j,k] = gps2dist_azimuth(lat,lon,lat_station[k],lon_station[k])[0]
                p = slowness_profile[np.argmin(np.abs(dist2source_matrix[i,j,k]-dist_profile))]  #en fonction de la distance on sait quelle slowness doit être utilisée pour corriger l'arrivée
                time_to_shift = p*(dist2source_matrix[i,j,k])/1000
                idx2shift_matrix[i,j,k] = int(np.round(time_to_shift*fs))
    return dist2source_matrix, idx2shift_matrix

def import_data(year,day,lon_station, lat_station,network_station, name_station, location_station, lowest_dist2coast_list,
                fs,filtered=None,freq=None, cooked=False):
    initial_station_number = len(name_station)
    print('Initial number of stations in the given range to the coast :', initial_station_number)
    if cooked==False:
        dat = h5py.File(f'/summer/faultscan/user/parisnic/m1_internship/parkfield/data_parkfield/data_{int(fs)}.0hz/daily/glob/{year}/day_{day:03}.h5','r')
    elif cooked==True:
        dat = h5py.File(f'/summer/faultscan/user/parisnic/m1_internship/parkfield/data_parkfield/data_{int(fs)}.0hz_cooked/daily/glob/{year}/day_{day:03}.h5','r')
    wiggle_matrix_full = np.zeros((len(name_station),int(3600*24*fs))) #on créé à la taille de toutes les stations du fichier, mais sera probablement réduit car manque data à certaines sta!!
    idx_nodata = []
    name_nodata = []
    for i, sta in enumerate(name_station):
        try: #on essaye de load et de filtrer
            wiggle = dat[f'{network_station[i]}'][f'{sta}.{location_station[i]}']['Z'][:]
            if filtered=='bandpass':
                b,a = butter(1, [freq[0],freq[1]], btype='bandpass', output='ba', fs=fs)
                wiggle = filtfilt(b,a,wiggle)
            wiggle_matrix_full[i,:] = wiggle
        except: #si on peut pas charger = pas de data, alors on retient l'indice et la station qui pose problème (pourra alors vérifier quelles stations ont fait chier!!
            idx_nodata.append(i)
            name_nodata.append(f'{network_station[i]}.{sta}')
    
    if len(idx_nodata)>0:
        wiggle_matrix_full = np.delete(wiggle_matrix_full, idx_nodata, axis=0)
        lon_station = np.delete(lon_station,idx_nodata)
        lat_station = np.delete(lat_station,idx_nodata)
        network_station = np.delete(network_station,idx_nodata)
        name_station = np.delete(name_station,idx_nodata)
        lowest_dist2coast_list = np.delete(lowest_dist2coast_list, idx_nodata)
        print(name_nodata, 'did not have any data!')
    else:
        print('Yey no data missing !!!')
    print(f'{len(name_station)} traces could be imported ({initial_station_number-len(name_station)} stations did not have data)')
    return wiggle_matrix_full, lon_station, lat_station,network_station, name_station,lowest_dist2coast_list


def remove_bad_station(wiggle_matrix_full, lon_station, lat_station,network_station, name_station,lowest_dist2coast_list,threshold):
    '''Fonction permettant de retirer les traces qui présentent à un quelquoncque moment un giga spike qui a une amplitude nettement trop important 
    -> puisuq'on se base sur moyenne ce spike vraiment un pb,  '''
    initial_station_number = len(name_station)
    name_removed = []
    idx_removed = []
    for i, sta in enumerate(name_station):
        if np.max(np.abs(wiggle_matrix_full[i,:]))/np.median(np.abs(wiggle_matrix_full[i,:]))>threshold:
            idx_removed.append(i)
            name_removed.append(f'{network_station[i]}.{sta}')
    if len(idx_removed)>0:
        wiggle_matrix_full = np.delete(wiggle_matrix_full, idx_removed, axis=0)
        lon_station = np.delete(lon_station,idx_removed)
        lat_station = np.delete(lat_station,idx_removed)
        network_station = np.delete(network_station,idx_removed)
        name_station = np.delete(name_station,idx_removed)
        lowest_dist2coast_list = np.delete(lowest_dist2coast_list, idx_removed)
        print(name_removed, 'had bad traces !')
    else:
        print('No stations were removed !!!') 
    print(f'{len(name_station)} stations left after removing bad ones ({initial_station_number-len(name_station)} removed).')
    return wiggle_matrix_full, lon_station, lat_station,network_station, name_station,lowest_dist2coast_list


def make_section_matrix(wiggle_matrix_full,wstart,wlen,fs):
    '''Recoupe la full matrix dans la window souhaitée'''
    idx_start = wstart*fs
    idx_end = idx_start+wlen*fs
    wiggle_matrix_section = wiggle_matrix_full[:,idx_start:idx_end]
    return wiggle_matrix_section

def slant_stack(wiggle_matrix,idx2shift_matrix,idx_lat,idx_lon,bin_list,mode='sum',bin_stack_envelope=False,norm_type='minmax',plt_shift=False, nth_root_stack=False):
    '''Calcule le slant stack des wiggles pour une position de  la grille réalisée  (donc au point idx_lat, idx_lon)
    nous la wiggle matrix qu'on lui donne c'est une window de la wiggle matrix full
    onva faire du slant stack mais dans chaque bin séparé et ensuite normaliser les bins avant de les stacker ensemble
    on peut aussi passer les binstack en enveloppe pour être sûr que les peaks se stackent de façon cohérent (utile pour la recherche de max)
    on peut aussi lui changer le norm_type  car le pb avec la normalisation c'est que ça squish les traces qui ont des pics donc ça squeesh les bin stack où il y a
    des arrivées qui ont bien été horizontalisées : -> gros problème car du coup la trace final a faible amplitude au lieu de forte amplitude'''
    number_of_bins = len(np.unique(bin_list)) # la bin liste est la liste de station = même size que wiggle matrix[:,0] avec leur numéro de bin pour savoir avec qui stacker!
    stack_matrix = np.zeros((number_of_bins,len(wiggle_matrix[0,:]))) 
    if plt_shift==True: #on définit alors les couleurs des divers bins, qui seront alors utilisées pour plot les wiggles et stacks !
        plt.figure(figsize=(10,6))
        color_coeff = 1/(np.max(bin_list)-np.min(bin_list))
        bin_list_colors = color_coeff*bin_list
        bin_list_colors = bin_list_colors-np.min(bin_list_colors)
        colors = plt.cm.tab20(bin_list_colors)
    for k in range(len(wiggle_matrix[:,0])): 
        wiggle_shifted = np.zeros(len(wiggle_matrix[0,:]))
        idx_to_shift = int(idx2shift_matrix[int(idx_lat),int(idx_lon),k])
        if idx_to_shift==0:
            wiggle_shifted = wiggle_matrix[k,:]
        else:
            wiggle_shifted[0:-idx_to_shift] = wiggle_matrix[k,idx_to_shift:]
        
        if plt_shift==True:
            plt.plot(np.arange(0,30*60,1/10),wiggle_shifted*0.5+dist2source_matrix[idx_lat,idx_lon,k]/1000, c=colors[k]) #attention source = pixel testé, pas la vraie source ofc
    
        bin_number = int(bin_list[k]) 
    
        if nth_root_stack==True: #on ajoute aussi la spossible de faire du nth root stack ! 
            signs = np.sign(wiggle_shifted)
            number_in_stack = np.count_nonzero(bin_list==bin_number) #compte le nombre de traces qui ont été stackées dans  
            stack_matrix[bin_number,:] += signs*(1/number_in_stack)*(signs*wiggle_shifted)**(1/4) #on fait déjà la moyenne dans la boucle
        else:
            stack_matrix[bin_number,:] += wiggle_shifted #on bien de faire juste du lienar stack     
    
    if nth_root_stack==True:#en dehors de la boucle on applique la fin du calcul sur les traces en dehors de la boucle, reste plus qu'à **4 le tout et multiplier par les signes
        signs = np.sign(stack_matrix) #récupère les signes de tous les stacks de la stack matrix
        stack_matrix = signs*stack_matrix**4 #met toute la matrix en **4 et remultiplie par les signes
        
    
    if bin_stack_envelope==True: #on a aussi la possibilité de passer les bin stacks en enveloppe -> assure que sommation des peaks cohérente entre les divers bins 
        for b, bin_stack in enumerate(stack_matrix): 
            analytic_signal = hilbert(bin_stack)
            stack_matrix[b,:] = np.abs(analytic_signal)
    
    if plt_shift==True: #on affiche toutes les traces shiftedpour pour une source lcoalisée à l'endroit indiqué
        plt.xlabel('time (s)')
        plt.ylabel('Distance to the station (km)')
        plt.title(f'Shifted traces for a potential source located at coordinates {(latitudes[idx_lat],longitudes[idx_lon])}')
        
        plt.figure() # à présent on fait la figure pour affichier les bin stacks
        color_coeff = 1/(np.max(bin_list)-np.min(bin_list))
        bin_list_colors = color_coeff*np.unique(bin_list) #on doit à présent utiliser que 1 fois chaque bin value car on a stack les membres du même bin! 
        bin_list_colors = bin_list_colors-np.min(bin_list_colors)
        colors = plt.cm.tab20(bin_list_colors)
        for b in range(number_of_bins):
            plt.plot(np.arange(0,30*60,1/10), stack_matrix[b,:]+b,c=colors[b])
        plt.xlabel('time (s)')
        plt.ylabel('bin number')
        plt.title('Different bin stacks before normalization')
    
    
    #on doit à présent renormaliser les bins 
    stack_matrix = traces_normalization(stack_matrix,norm_type=norm_type) #on a renormalisé les bin stacks   
    stack = np.sum(stack_matrix,axis=0) #on stack à présent les bin stacks 
    
    if bin_stack_envelope==False:
        # à présent, on fait la hilbert transform du slant stack, pour avoir son enveloppe et donc avoir un truc un peu plus smooth à voir si on garde....
        analytic_signal = hilbert(stack) #peut être qu'on pourrait faire hilbert transform sur les bin stacks avant de faire le stack final ? 
        stack = np.abs(analytic_signal)
    
    if plt_shift==True: #on plot aussi les stacks normalisés pour les diverses stations
        plt.figure() # à présent on fait la figure pour affichier les bin stacks
        color_coeff = 1/(np.max(bin_list)-np.min(bin_list))
        bin_list_colors = color_coeff*np.unique(bin_list) #on doit à présent utiliser que 1 fois chaque bin value car on a stack les membres du même bin! 
        bin_list_colors = bin_list_colors-np.min(bin_list_colors)
        colors = plt.cm.tab20(bin_list_colors)
        for b in range(number_of_bins):
            plt.plot(np.arange(0,30*60,1/10), stack_matrix[b,:]+b,c=colors[b])
        plt.xlabel('time (s)')
        plt.ylabel('bin number')
        plt.title('Different bin stacks after normalization')
    
        plt.figure() #à présent on veut afficher le stack résultant
        plt.plot(np.arange(0,30*60,1/10),np.abs(stack), label='abs(stack)') #de base c'était juste abs de stack qu'on utilisait pour faire la somme, mais now utilise envloeppe
        # analytic_signal = hilbert(stack)
        # plt.plot(np.arange(0,30*60,1/10),np.abs(analytic_signal), label='enveloppe of stack')
        # plt.legend()
        plt.xlabel('time (s)')
        plt.ylabel('amplitude (vel)')
        plt.title('Stack')    
    
    if mode=='mean':                #maintenant que le stack est fini, on calcule soit la somme, soit la moyenne, soit la médiane du stack pour avoir une amplitude de pixel; cette 
        amp = np.mean(np.abs(stack))#amplitude de pixel sera encore sommée avec les 47 autres de la journée pour faire un stack global!!
    elif mode=='sum':
        amp = np.sum(np.abs(stack))
    elif mode=='median':
        amp = np.median(np.abs(stack))
    elif mode=='max':
        amp = np.max(np.abs(stack))
    elif mode=='inverse_sum':
        amp = 1/np.sum(np.abs(stack)) #si bin stacké de façon cohérente alors spikes => donc toute la trace écrasée => amplitude de la trace faible => 
        # => donc somme de traces écarsées => trace de faible amplitude as well => faible amplitude = zone où se trouve la source !
    return amp #return un float seulement


def traces_normalization(trace_matrix,verbose=0,norm_type='minmax'):
    trace_matrix_normalized = np.zeros_like(trace_matrix)
    for i in range(len(trace_matrix[:,0])):
        if norm_type=='minmax':
            coeff = 1/(np.max(trace_matrix[i,:]-np.min(trace_matrix[i,:]))) 
        elif norm_type=='std':
            coeff = 1/(4*np.std(trace_matrix[i,:]))
        trace_matrix_normalized[i,:] = coeff*trace_matrix[i,:]
    if verbose>0:
        print('Normalization of every trace done')
    return trace_matrix_normalized


def compute_dist2coast_list(lat_sta,lon_sta):
    '''Fonction permettant de calculer la distance à la côte de toutes les stations; elle est utile pour savoir quelles stations garder
    et pour faire du bnning aka regrouper les stations ensemble par distance à la côte'''
    lowest_dist2coast_list = np.zeros(len(lat_sta))
    coords_coast = np.load('coords_coast.npy')
    lat_coast, lon_coast = coords_coast[:,1], coords_coast[:,0]
    for i in range(len(lat_sta)):
        lowest_dist2coast = 9999999999999999999999999999999999999999
        for j in range(len(lat_coast)):
            dist2coast = gps2dist_azimuth(lat_sta[i], lon_sta[i], lat_coast[j], lon_coast[j])[0]/1000 #calcule de la distance de la station à un pt de la coast en km!!
            if dist2coast<lowest_dist2coast:
                lowest_dist2coast = dist2coast
        lowest_dist2coast_list[i] = lowest_dist2coast #on retient alors la plus faible distance vis à vis de la coast de cette station
    print('Distance of stations from coastline successfully computed')
    return lowest_dist2coast_list #on connait à présent la distance des stations à la côte
    


def binning(lat_station, lon_station, latitude_ticks,longitude_ticks):
    '''Retuns une liste qui indique le numéro de bin de chacune des stations : le binning se fait selon une grille de coordonnées
    pour la binlist : on liste     
    attention le step de lat et de lon doit du coup être en degré et non en km comme c'était le cas pour le binning selon la distance à la coastline'''
    bin_station_list = np.zeros(len(lat_station))
    for i in range(len(bin_station_list)):
        idx_lat = np.argmin(np.abs(latitude_ticks-lat_station[i]))
        idx_lon = np.argmin(np.abs(longitude_ticks-lon_station[i]))
        bin_station_list[i] = idx_lat*len(longitude_ticks)+idx_lon #on donne un numéro de cellule 
    
    bin_indexes = np.unique(bin_station_list) #returns the indexes of the different bins and len gives us number of bins 
    for i in range(len(bin_indexes)):
        bin_station_list[np.where(bin_station_list==bin_indexes[i])] = i #on remplace les index par des bin numbers sans coupure
    print('Binning done!')
    return bin_station_list


def plot_bins(bin_list,lon_station,lat_station, latitude_ticks, longitude_ticks):
    '''Fonction juste pour le notebook permettant de montrer les bins réalisés!   Permet par la même occasion de plot les cellules permettant de vérifier que c'est bien rassemblé
    !!'''
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    extent = [np.min(lon_station)-1, np.max(lon_station)+1, np.min(lat_station)-1, np.max(lat_station)+1]
    ax.set_extent(extent)
    stamen_terrain = Stamen(desired_tile_form="RGB", style="terrain-background") 
    ax.add_image(stamen_terrain, 8)
    ax.coastlines()        
    
    color_coeff = 1/(np.max(bin_list)-np.min(bin_list))
    bin_list_colors = color_coeff*np.unique(bin_list) #on fait en sorte d'avoir le nombre minimal de ticks nécessaires!
    bin_list_colors = bin_list_colors-np.min(bin_list_colors)
    colors = plt.cm.tab20(bin_list_colors)
    
    
    for i,BIN in enumerate(np.unique(bin_list)):
        stations_in_BIN = np.where(bin_list==BIN)
        ax.scatter(lon_station[stations_in_BIN], lat_station[stations_in_BIN],c=colors[i], marker='^', transform=ccrs.PlateCarree(), s=30)
    
    for i in range(len(bin_list)):
        ax.text(lon_station[i], lat_station[i], f'{int(bin_list[i])}')
    
    #on plot à présent les lignes correspondant aux cellules dans lesquelle sont cherche à avoir les stations
    lon_step = longitude_ticks[1]-longitude_ticks[0]
    lat_step = latitude_ticks[1]-latitude_ticks[0]
    for i in range(len(longitude_ticks)):
        if  longitude_ticks[i]+lon_step/2>= extent[0] and longitude_ticks[i]<= extent[1]:
            plt.plot([longitude_ticks[i]+lon_step/2, longitude_ticks[i]+lon_step/2], [extent[2], extent[3]],
                 color='red', linewidth=0.7,transform=ccrs.PlateCarree())  #on décale la ligne de la moitié du step histoire d'être OK !
    for i in range(len(latitude_ticks)):
        if  latitude_ticks[i]+lat_step/2>= extent[2] and latitude_ticks[i]<= extent[3]:
            plt.plot([extent[0],extent[1]], [latitude_ticks[i]+lat_step/2, latitude_ticks[i]+lat_step/2],
                 color='red', linewidth=0.7,transform=ccrs.PlateCarree())  #on décale la ligne de la moitié du step histoire d'être OK !
    
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0, color='gray', alpha=0.5, linestyle='--')
    gl.bottom_labels = False
    gl.right_labels = False

    
    
def select_range2coast(station_file, distance_range):
    '''Première fonction, elle permet de sélectionner les stations dans la range elle return alors les metadata sur les stations restantes
    + les distances à la coastline, ce qui est utilisé plus tard pour le binning as well!'''
    lon_station, lat_station, heights_station,network_station, name_station, location_station  = extract_coordinates(station_file, sta_name=True, sta_loc=True)
    #on a importé le fichier avec les metadata des stations, on cherche à présent à vérifier si elles sont dans range à la coast
    lowest_dist2coast_list = compute_dist2coast_list(lat_sta=lat_station, lon_sta=lon_station)
    #à présent on itère dsur la liste de distances à la coast pour savoir si on respecte la range 
    idx_notinrange = []
    name_notinrange = []
    for i, lowest_dist2coast in enumerate(lowest_dist2coast_list):
        if lowest_dist2coast<distance_range[0] or lowest_dist2coast>distance_range[1]:
            idx_notinrange.append(i)
            name_notinrange.append(f'{network_station[i]}.{name_station[i]}')
    if len(idx_notinrange)>0:
        lon_station = np.delete(lon_station,idx_notinrange)
        lat_station = np.delete(lat_station,idx_notinrange)
        network_station = np.delete(network_station,idx_notinrange)
        name_station = np.delete(name_station,idx_notinrange)
        location_station = np.delete(location_station, idx_notinrange)
        lowest_dist2coast_list = np.delete(lowest_dist2coast_list, idx_notinrange)
        print(f'{name_notinrange} are not in the provided ({distance_range}) to the coastline!')
    else:
        print('All stations were in the given range to the coast!!!')
    return lon_station, lat_station,network_station, name_station, location_station, lowest_dist2coast_list
    
    
def compute_daily_slant_stack(year,day,slowness_file,station_file,longitudes,latitudes,latitude_ticks,longitude_ticks,filtered,freq,fs,wlen,mode='sum',cooked=False,rm_bad_station=False,
                              threshold=999, distance_range=[0,999],bin_stack_envelope=False,norm_type='minmax',nth_root_stack=False, rm_handselected=None):
    '''Latitud eticks et longitude ticks c'est pour le binning  on doit donenr des arrays correspondant aux bins, comme ça c'est indépendant de la zone dans laquelle on regarde
    dans la carte finale : toutes les stations peuvent tout de même être correctement binned!'''
    #on commence par sélectionner les stations qui sont dans la range de distance à la côte
    lon_station, lat_station,network_station, name_station, location_station, lowest_dist2coast_list = select_range2coast(station_file, distance_range)
    
    #on importe les données directemenent des stations dans la range 
    wiggle_matrix_full, lon_station, lat_station,network_station, name_station,lowest_dist2coast_list = import_data(year=year,day=day,
    lon_station=lon_station, lat_station=lat_station,network_station=network_station, name_station=name_station,location_station=location_station,
    lowest_dist2coast_list=lowest_dist2coast_list,fs=fs,filtered=filtered,freq=freq,cooked=cooked)
    
    if rm_bad_station==True:
        #focntion pour remove les bad stations : crtière genre si le max de la trace est -> shifter ça dans la fonction d'import ??? 
        wiggle_matrix_full, lon_station, lat_station,network_station, name_station,lowest_dist2coast_list = remove_bad_station(wiggle_matrix_full=wiggle_matrix_full,
        lon_station=lon_station,lat_station=lat_station,network_station=network_station, name_station=name_station,lowest_dist2coast_list=lowest_dist2coast_list,
        threshold=threshold)
    
    if rm_handselected!=None: #fonctoion pour remove les badstations sélectionnées à la main : aurait pu être mis tout au début tbh
        wiggle_matrix_full, lon_station, lat_station,network_station, name_station,lowest_dist2coast_list = remove_handselected_stations(wiggle_matrix_full, lon_station,
                                                        lat_station,network_station, name_station,lowest_dist2coast_list,rm_handselected)
    
    #on a présent removed les stations en dehors de la range, rremoved les stations sans data et celles avecmauvaises data : on détermine alors pour chaque station restante
    #sa distance vis à vis des pixels de la carte ainsi que les slowness alors attendues pour les arrivées issues de source à ces pixels
    dist_profile, slowness_profile = np.loadtxt(slowness_file) #on a amitnenant import nos stations et notre profil de slowness : on peut calculer 
    dist2source_matrix, idx2shift_matrix = compute_dist2source_idxshift_matrix(longitudes,latitudes,lon_station,lat_station,dist_profile,slowness_profile,fs)  #la matrix d'idx to shift 
    print('Distances and timeshift for every pixel-station combination computed') #attention peut pas sortir matrix shift d'ici car pas forcémrnt les même stations les diff jours
    
    #on fait à présent du binning pour savoir comment doivent être regroupées et traces et ainsi savoir combien de bins vont être créés et savoir dans quel bin chacun se trouve
    bin_list = binning(lat_station=lat_station, lon_station=lon_station, latitude_ticks=latitude_ticks, longitude_ticks=longitude_ticks) #on bin alors les stations ce qui servira à
    wstart_list = np.arange(0,86400,wlen) #on créé la liste des wstart qui serviront alors   
    amp_matrix = np.zeros((len(latitudes),len(longitudes))) #on créé la hmap, à chaque window on stack la nouveelle hmap avec la précédente
    
    wiggle_matrix_full = traces_normalization2coast(wiggle_matrix_full=wiggle_matrix_full,lowest_dist2coast_list=lowest_dist2coast_list,fs=fs) # on normalize à présent distance par la distance à la côte
    print('Computing slant stacks for every window, please wait...')
    for wstart in tqdm(wstart_list): #on loop sur les windows 
        amp_matrix_temp = np.zeros((len(latitudes),len(longitudes)))
        wiggle_matrix_section = make_section_matrix(wiggle_matrix_full=wiggle_matrix_full,wstart=wstart,wlen=wlen,fs=fs) #on est dans une window donnée donc on recoupe la matrice en fonction de ça 
        # wiggle_matrix_section = traces_normalization(trace_matrix=wiggle_matrix_section,verbose=0, norm_type='minmax') # DEPRECATED WAS FOR NORMALIZE MIN MAX 
        for i, lat in enumerate(latitudes):
            for j, lon in enumerate(longitudes):
                amp_matrix_temp[i,j] = slant_stack(wiggle_matrix=wiggle_matrix_section,idx2shift_matrix=idx2shift_matrix,idx_lat=i,idx_lon=j,
                                                   bin_list=bin_list,mode=mode,bin_stack_envelope=bin_stack_envelope,norm_type=norm_type,plt_shift=False,
                                                  nth_root_stack=nth_root_stack)
        amp_matrix += amp_matrix_temp #stacking final
    return amp_matrix, lat_station, lon_station #il rééexporte lon et lat station car on en a retiré en fonction de si y'a data + du QC


def plot_daily_slant_stack(year,day,latitudes,longitudes,amp_matrix, lat_station,lon_station, save_directory):
    '''Fonction permettant de plotter carte sur de l'amplitude matrix  générée par le slant stack pour toutes les positions de source potentielles !!!'''
    plt.close('all')
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    extent = [longitudes[0], longitudes[-1], latitudes[0], latitudes[-1]]
    ax.set_extent(extent)
    stamen_terrain = Stamen(desired_tile_form="RGB", style="terrain-background") 
    ax.add_image(stamen_terrain, 8)
    ax.coastlines()        
    hmap = ax.pcolormesh(longitudes, latitudes, amp_matrix,
         transform=ccrs.PlateCarree(), cmap='jet',alpha=0.4, vmin=np.min(amp_matrix), vmax=np.max(amp_matrix))
    cbar = plt.colorbar(hmap)
    ax.scatter(lon_station, lat_station, color='red', marker='^', transform=ccrs.PlateCarree(), s=30)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0, color='gray', alpha=0.5, linestyle='--')
    gl.bottom_labels = False
    gl.right_labels = False
    
    plt.text(0.4,-0.08,f'{datetime(year,1,1)+timedelta(day-1)}',transform=ax.transAxes)
    
    plt.savefig(f'{save_directory}gridsearch_map_{year}_day_{day:03}.png', dpi=300, bbox_inches='tight')
    
    
def plot_stations(station_file,lon_station,lat_station):
    '''Fonction juste pour le notebook permettant de montrer les stations totales et celles sélectionnées parmis le sstations globales
    le station file lui permet de savoir où sont toutes les stations, tandis que lat et lon station permet de plotter celles qui sont réellement sélectionnées
    cette focntion peut être utilisée pour plot àtoutes les étapes de la sélection donc dans la range, celles avec data et les bad ones!'''
    lon_station_full,lat_station_full, heights_station_full = extract_coordinates(station_file)
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    extent = [np.min(lon_station_full)-1, np.max(lon_station_full)+1, np.min(lat_station_full)-1, np.max(lat_station_full)+1]
    ax.set_extent(extent)
    stamen_terrain = Stamen(desired_tile_form="RGB", style="terrain-background") 
    ax.add_image(stamen_terrain, 8)
    ax.coastlines()        
    
    ax.scatter(lon_station_full, lat_station_full, color='red', marker='^', transform=ccrs.PlateCarree(), s=30)
    ax.scatter(lon_station, lat_station, color='green', marker='^', transform=ccrs.PlateCarree(), s=30)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0, color='gray', alpha=0.5, linestyle='--')
    gl.bottom_labels = False
    gl.right_labels = False
    
    
    

def remove_handselected_stations(wiggle_matrix_full,lon_station,lat_station,network_station,name_station,lowest_dist2coast_list, rm_handselected):
    print(f'There are {len(lon_station)} stations before removing handselected stations ({len(rm_handselected)} stations to remove')  
    idx_removed = []
    name_removed = []
    for i in range(len(rm_handselected)):
        find = np.where((network_station==rm_handselected[i][0]) & (name_station==rm_handselected[i][1]))
        try:
            idx_removed.append(find[0][0])
            name_removed.append(f'{rm_handselected[i][0]}.{rm_handselected[i][1]}')
        except:
            pass
    wiggle_matrix_full = np.delete(wiggle_matrix_full, idx_removed, axis=0)
    lon_station = np.delete(lon_station,idx_removed)
    lat_station = np.delete(lat_station,idx_removed)
    network_station = np.delete(network_station,idx_removed)
    name_station = np.delete(name_station,idx_removed)
    lowest_dist2coast_list = np.delete(lowest_dist2coast_list, idx_removed)
    print(name_removed, 'were removed by hand!')
    return wiggle_matrix_full, lon_station, lat_station,network_station, name_station,lowest_dist2coast_list


def traces_normalization2coast(wiggle_matrix_full,lowest_dist2coast_list,fs=10):
    '''Cette fonction permet de normaliser non pas selon indice commun mais bien de normaliser les traces en focntion de la distance à la coastline'''
    wiggle_matrix_normalized = np.zeros_like(wiggle_matrix_full)
    wiggle_matrix = make_section_matrix(wiggle_matrix_full,30*60,79200,fs) #on doit lui donner la wiggle matrix full pour ressortir la wiggle matrix full normalized !!!
    traces_energy = 1/len(wiggle_matrix[0,:])*np.sum(np.abs(wiggle_matrix), axis=1)
    fit = np.polyfit(lowest_dist2coast_list, np.log10(traces_energy), 2) 
    curve = 10**(np.polyval(fit, lowest_dist2coast_list))
    for i in range(len(wiggle_matrix_full[:,0])):
        wiggle_matrix_normalized[i,:] = wiggle_matrix_full[i,:]/curve[i]
    print('Normalization of every trace done according to their distance to the coast') #on fait la normalisation sur la journée entière pour avoir stat robuste par conséquent on peut se permettre de print car pas dans loop!
    return wiggle_matrix_normalized
