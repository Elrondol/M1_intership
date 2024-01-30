# This script will check available channels on FDSN catalog
# OUtputs are a station list as a light .txt files, a full .pkl file 
# and a station map .png
# 
# OUTPUT station file structure :
# datacenter   net   sta   locid   lat     lon      alt     depth
# RESIF        YP    CT21  00      44.654  6.56859  1171.0  0.0
#
########################################################
######################################################## 
########################################################  
import m_pycorr.m00_get_stations as get_sta
from obspy.core import UTCDateTime
in_                   = {}

# Here you'll find all possible options and their format 
# =====>    http://service.iris.edu/irisws/fedcatalog/1/
#in_['key']        = value 

stafile                = 'stations_' # output name (.txt and .png)

###############  channel ID

in_['net']                 = '*' # 'net1,net2,net2...' accept wildcard
in_['sta']                 = '*' # 'sta1,sta2,sta3...' accept wildcard
in_['loc']                 = '*' # 'loc1,loc2,loc3...' accept wildcard
in_['channel']             = 'BHZ,HHZ,EHZ' # 'ch1,ch2,ch3...' accept wildcard
in_['includeavailability'] = True  # Specify if results should include information
                                   # about time series data availability at the channel level.
###############  date selection

in_['startbefore']     = '2002-01-01' # yyyy-mm-dd
in_['endafter']        = '2002-01-01' # yyyy-mm-dd
#in_['start']           = '2013-04-16' # yyyy-mm-dd
#in_['end']             = '2013-04-30' # yyyy-mm-dd

###############  for a circular search :

in_['lat']             = '35.724775067472635' # center point lat
in_['lon']             = '-121.28399495555139' # center point lon
in_['maxradius']       = '1.5' # max radius
in_['minradius']       = '0' # min radius

###############  for a rectangular search :

#in_['minlat']          = '45' # top
#in_['maxlat']          = '0' # bottom
#in_['minlon']          = '10' # right/left
#in_['maxlon']          = '0' # left/right

# ... other parameters are available on http://service.iris.edu/irisws/fedcatalog/1/

############### RUN
get_sta.find_stations(input_user=in_,stafile=stafile)

