# This script will start the download of a specific path 
# /data_*freq*Hz/daily/**tag**/**year**/
#
# This script can be run multiple times
# A natural parallelization will be done a 
# daily basis using temproray *.lock or *.cplt files
#
# OUTPUT DATA WILL BE STORED IN H5 FILES
#
# day_XXX.h5 |- /net1|-sta1.locid|-ch1 (dataset)
#            |       |           |-chX (dataset)
#            |       |           |-... (dataset)
#            |       |             
#            |       |-sta2.locid ...
#            |       |-...
#            |       |-staX.locid ...
#            |
#            |- /net2 ...
#            |- ...
#            |- /netX ...
#            |- /metadata|-fe (dataset)
#                        |-t0_UNIX_timestamp (dataset)
#                        |-sta (group) ...metadata for all downloaded stations
#
########################################################
######################################################## 
######################################################## 
import m_pycorr.m10_get_data as get_data
in_ = {}

############### Standard Inputs
in_['path']='./data_40.0hz/daily/glob/2002' # directory to run, 
in_['mode']= 0

#   -- mode = 0 : Initial download: Do not require _metadata directory (containing metadata)
#   -- mode = 1 : add Data: attempt to download ALL missing data in the h5 file as defined in db.pkl from config. Do not require _metadata
#   -- mode = 2 : complete: re-download all (1) new stations, and data that were canceled (2) because the datacenter could not be reached or because of a timeout
#   -- mode = 3 : same as 2 but also for rejected traces

# WARNING:
# remember to remove potential temporary files (*.lock, *.cplt) and their corresponding *.h5 files form your downloading archive if something goes wrong and you need to re-run.

############### RUN
get_data.download(in_)


