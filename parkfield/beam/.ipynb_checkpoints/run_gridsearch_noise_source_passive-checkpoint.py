import numpy as np 
from tqdm import tqdm
from functions import * 


#######################" PARAMETRE A CHANGER  OFC ##############
year = 2002
days = range(275,366)

#version intégrale
# longitudes = np.linspace(-124.0,-118.4,100)
# latitudes = np.linspace(33.6, 39.2, 100)

#zoom 
longitudes = np.linspace(-122.6,-120.4,100)
latitudes = np.linspace(35.2, 36.2, 100)
################################################################

###### PARAMETERS PRETTY MUCH GOOD TO GO WITH THAT
filtered = 'bandpass'
freq = [0.83,1.33]
fs = 10  #utilise 20 hz = d'autres stations en + 
wlen = 30*60 #durée de la window en secondes
mode='sum' #l'intensité du pixel est la sum de la valeur absolue du stack (ou plutôt de la valeur absolue de l'enveloppe)
station_file = 'stations_slant_moresta.txt' #utilise le dataset adapté pour data cooked (doit en retirer si uncooked donc autre station list)
cooked = True  #cooked donc pas de clipping
rm_bad_station = True
threshold = 100 #si la trace présente un pic avec threshold fois l'amplitude de la médiane alors on la retire de la liste
save_directory = f'/summer/faultscan/user/parisnic/m1_internship/parkfield/MAP_SOURCE_FIGS/slant_stack/'  


print('Working on day, please wait ...')
for day in tqdm(days):
    amp_matrix,lat_station,lon_station = compute_daily_slant_stack(year=year,day=day,station_file=station_file, longitudes=longitudes,latitudes=latitudes,filtered=filtered,
                                           freq=freq,fs=fs,wlen=wlen,mode=mode,cooked=cooked, rm_bad_station=rm_bad_station,threshold=threshold)
    plot_daily_slant_stack(year=year, day=day, latitudes=latitudes,longitudes=longitudes,amp_matrix=amp_matrix,lat_station=lat_station,
                           lon_station=lon_station, save_directory=save_directory)