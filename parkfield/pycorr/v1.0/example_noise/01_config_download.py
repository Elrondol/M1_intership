# This script will create a directory containing a db.pkl file 
# This db.pkl contains all information to download the data
# 
########################################################
######################################################## 
########################################################  
import m_pycorr.m10_get_data as get_data
from obspy.core import UTCDateTime
in_                   = {}

############### Standard Inputs

in_['ch']             = ['LH'] # ['ch1','ch2',...] It will try in this order. Better to start from the lower sampling rate ['LH','BH','HH'...]
in_['cmp']            = ['Z'] # ['cmp1','cmp2',...]
in_['tag']            = 'glob' # subdirectory name for the output /data_*freq*Hz/dauly/**tag**/**year**/
in_['station_list']   = 'stations.txt' # input station file from 00_get_stations.py or other
in_['path_out']    = './' # output path

############### conventional preprocessing

in_['pp']                = {} # always '{}'
in_['pp']['freq']        = 1. # [hz], freq. to which data are decimated [hz] => output sampling rate
in_['pp']['f_prefilt']   = (0.005, in_['pp']['freq'] / 2 - 0.05 * in_['pp']['freq']) # [hz], (f0,f1) used for butter-BP filter after sensor response deconvolution
in_['pp']['tap_len']     = 15*60 # [s], dwld date1-tap_len -> date2+tap_lens but keep date1 -> date2, NOT valid for SDS archive !
in_['pp']['glitch']      = False # True/False, remove or not glitches larger than 20 std, replace by 0 
in_['pp']['remove_resp'] = True #  True/False, If False, it will try to download resp from clients but DO NOT try to read from SDS archive
    
############### QC

in_['qc']                  = {} # always '{}'
in_['qc']['max_nsegments'] = 12 # trash trace if more than x segments
in_['qc']['min_duration']  = 72000 # [s], trash trace if length than x sec
in_['qc']['timeout']       = 300 # [s] walltime in sec for processing 1 trace


############### if dc == 'SDS'

# loading from (formated) 
# directory instead of online datacenters.
# see m10_get_data() and read_SDS() function for details
#in_['sds_path']       = 'path_to_your_data/year/net/ccmp.D/net.sta.loc.ccmp.D.year.jday' #(SAC, MSEED ...)
#in_['sds_metapath']   = 'path_to_your_metadata/net/net.sta.station.response.xml' # .xml only !

############### if data restriction on datacenters

#in_['user_id']        = None # user id for fdsn client
#in_['password']       = None # password for fdsn client


############### RUN FOR DATES
list_year                  = [2018] # list of targeted year

for iyear in list_year : # loop over a range of dates
    in_['day1']=UTCDateTime(iyear,4,16) # date 1
    in_['day2']=UTCDateTime(iyear,5,16) # date 2
    get_data.configure(in_)
