from functions import *
from variables import *

sdate = (2001,12,20)
edate = (2001,12,21)
step = '12h'

#coords pour noise 
#longitudes = np.linspace(-121.8,-120.4,40) #coordonnées pour  le noise
#latitudes = np.linspace(35.5, 36.1,30)
#station_file = 'stations_rm_nobad.txt'
#s_depth = 0.0


#coords pour avenal
longitudes = np.linspace(-119.8,-121.0,40)
latitudes = np.linspace(35.6, 36.3, 30)
station_file = ''
s_depth = 8.8


#coords pour eq gonzales
#latitudes = np.linspace(35.6, 37.2, 30)  #coords pour gonzales
#longitudes = np.linspace(-122.5,-120.4,40)
#station_file = 'stations_rm_nobad_gonzales.txt'
#s_depth = 6.8


    
badbeam_list = [datetime(2001,10,6,12),datetime(2001,10,7),datetime(2001,10,7,12),  #jours 279.5, 280 et 280.5
                datetime(2001,10,22,12),datetime(2001,10,23),datetime(2001,10,23,12), #jours atours de 296
                datetime(2001,11,15,12),datetime(2001,11,16),datetime(2001,11,16,12), #jours autour 320
                datetime(2001,11,20,12),datetime(2001,11,21),datetime(2001,11,21,12), #jours autour 325
                datetime(2001,11,26,12),datetime(2001,11,27),datetime(2001,11,27,12), #jours autour 331
                datetime(2001,12,30,12),datetime(2001,12,31), #jour autour de 365
                datetime(2002,2,4,12), datetime(2002,2,5), datetime(2002,2,5,12), # jour 36
                datetime(2002,2,12,12), datetime(2002,2,13), datetime(2002,2,13,12), # jour 44
                datetime(2002,4,13), datetime(2002,4,13,12), # jour 103 et 103.5
                datetime(2002,4,28,12),datetime(2002,4,29),datetime(2002,4,29,12), #jour 119
                datetime(2002,5,1,12),datetime(2002,5,2),datetime(2002,5,2,12),datetime(2002,5,3),datetime(2002,5,3,12), #début mai 
                datetime(2002,5,4,12),datetime(2002,5,5),datetime(2002,5,5,12), #jour 125
                datetime(2002,5,9,12),datetime(2002,5,10),datetime(2002,5,10,12), #jour 130
                datetime(2002,5,23),datetime(2002,5,23,12), #jour 143 et 143.5
                datetime(2002,5,24,12),datetime(2002,5,25),datetime(2002,5,25,12),datetime(2002,5,26),datetime(2002,5,26,12) #fin mai 144.5-146.5
                ]    

dates_range, Z_mesh = compute_backpropag(sdate, edate, step,latitudes, longitudes, badbeam=badbeam_list,v_model='slowness_avenal_new_resampled', s_depth=s_depth, plot_map=False,p_option='fw',station_file=station_file)

Z_mesh_reshaped = Z_mesh.reshape(Z_mesh.shape[0], -1)

np.savetxt('Z_mesh_fw_new_avenal', Z_mesh_reshaped)
