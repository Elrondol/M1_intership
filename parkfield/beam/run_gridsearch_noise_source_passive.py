import numpy as np 
from tqdm import tqdm
from functions import * 
import os

#######################" PARAMETRE A CHANGER  OFC ##############
folder = 'test'
year = 2002
days = range(60,61)
distance_range = [0,200]
binstep = 0.25
latitude_ticks = np.arange(-124.0,-118.4,binstep) 
longitude_ticks = np.arange(33.6, 39.2,binstep)
mode = 'inverse_sum' # sum or inverse_sum or max
bin_stack_envelope=False 
norm_type = 'minmax' #minmax or std   #normalisation affecte que les bins stacks; les traces de bases sont normalsiées en min max  / distance 2 coast 
nth_root_stack = True


#version intégrale
#longitudes = np.linspace(-124.0,-118.4,40)
#latitudes = np.linspace(33.6, 39.2, 40)

#zoom 
longitudes = np.linspace(-123.0,-120.4,30)
latitudes = np.linspace(35.2, 36.2, 30)
################################################################

###### PARAMETERS PRETTY MUCH GOOD TO GO WITH THAT
filtered = 'bandpass'
freq = [0.83,1.33]
fs = 10  #utilise 20 hz = d'autres stations en + 
wlen = 30*60 #durée de la window en secondes
station_file = 'stations_slant_moresta.txt' #utilise le dataset adapté pour data cooked (doit en retirer si uncooked donc autre station list)
slowness_file = 'slowness_gonzales_resampled_smoothed'
cooked = True  #cooked donc pas de clipping
rm_bad_station = True
threshold = 100 #si la trace présente un pic avec threshold fois l'amplitude de la médiane alors on la retire de la liste
rm_handselected = [['XN','BUZZ'], ['XN','VINE'], ['XN','MRED'], ['XN','CRAB'], ['WR','SCK'],['NC','PWM']]

###################################################

folder_full_name = f'{folder}_range_{distance_range[0]}-{distance_range[1]}_binstep_{binstep}_mode_{mode}_BSE_{bin_stack_envelope}_norm_{norm_type}_nthrootstack_{nth_root_stack}_norm2coast'
save_directory = f'/summer/faultscan/user/parisnic/m1_internship/parkfield/MAP_SOURCE_FIGS/{folder_full_name}/'  

try:
    os.mkdir(save_directory)
except:
    print('Directory already exists, watch out!')


print('Working on day, please wait ...')
for day in tqdm(days):
    amp_matrix,lat_station,lon_station = compute_daily_slant_stack(year=year,day=day,slowness_file=slowness_file,station_file=station_file, longitudes=longitudes,
                                    latitudes=latitudes,latitude_ticks=latitude_ticks, longitude_ticks=longitude_ticks,filtered=filtered,
                                    freq=freq,fs=fs,wlen=wlen,mode=mode,cooked=cooked, rm_bad_station=rm_bad_station,threshold=threshold, distance_range=distance_range,
                                    bin_stack_envelope=bin_stack_envelope, norm_type=norm_type, nth_root_stack=nth_root_stack, rm_handselected=rm_handselected)


    plot_daily_slant_stack(year=year, day=day, latitudes=latitudes,longitudes=longitudes,amp_matrix=amp_matrix,lat_station=lat_station,
                           lon_station=lon_station, save_directory=save_directory)
