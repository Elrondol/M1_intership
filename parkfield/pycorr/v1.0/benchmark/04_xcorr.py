# This script will run cross-correlation between all pairs using
# xcorr.xcorr_all() function.
#
# This script can be run multiple times
# A natural parallelization will be done on a 
# daily basis using temproray *.lock or *.cplt files
# AND on a tag AND file size basis
# 
# OUTPUT FILE STRUCTURE:
#
#C1_04_xcorr_tag_cc_func/
#   daily  files : xcorr_tag1_tag2_000/yyyt_mm_01.h5
#                                     /yyyt_mm_02.h5
#                                     /yyyt_mm_dd.h5
#                   xcorr_tag1_tag2_001/...
#                   xcorr_tag1_tag2_XXX/...
#   master files : xcorr_tag1_tag2_XXX.h5
#
#  GROUPS inside H5 files
#   cc                       daily/hourly/... cross-correlations tables 
#   cc_nstack                order of summation for normalization 
#   in_                      input parameters passed to xcorr.xcorr_all() 
#   md                       metadata for paths (lat,lon, ...)
#   md_c                     metadata for correlations
#   ref                      STACKED CROSS_CORRELATIONS
#   ref_nstack               order of summation for normalization 
#
# ...XXX.h5|- /cc  |- net1.sta1.locid |- net1.sta1.locid |- cmp1 (dataset,table)
#          |       |                  |                  |- cmp2 (dataset,table)
#          |       |                  |                  |- cmp3 (dataset,table)
#          |       |                  |                  |- ...
#          |       |                  | 
#          |       |                  |- net1.sta2.locid |- ...
#          |       |                  |- net1.stax.locid |- ...
#          |       |                  |- nety.staz.locid |- ...
#          |       | 
#          |       |- net1.sta2.locid
#          |       |- netx.stay.locid
#          |                
#          |- /ref | same as cc ... (dataset,array)
#          |- /other groups
#
########################################################
######################################################## 
########################################################
import m_pycorr.m30_xcorr as xcorr 
from obspy.core import UTCDateTime
import numpy as np
in_ = {}

############### PATH(S)
in_['path']              = ['./data_4.0hz_cooked/daily/cifalps'] # list of path to correlate ['../daily/tag1,', '../daily/tag2']
in_['path_out']          = './' # output path
in_['tag']               = 'cifalps_conti' # output label (opt)
in_['file_size']         = 1. # [Gb] maximum size of each final h5 file, an good way to increase parallelization 

############## DATES
date1 = int(UTCDateTime(2013,4,16,0,0,0).timestamp) # from 
date2 = int(UTCDateTime(2013,4,26,0,0).timestamp) # to
in_['date']              = range(date1,date2+1,86400)
in_['hr1']               = list(np.linspace(0,24-1,24)) # if so  hour of the beginning of each segments 
in_['hr2']               = list(np.linspace(1,24,24)) # hour of the end of each segment

############## CC PARAMS
in_['cc_maxlag']         = 300. # [s], correlations will corresponds to cc_maxlag*2*fe+1 samples 
in_['cc_cmp']            = ['ZZ','NN','EE','NE','EN']  # list of channels combination to correlate. 
in_['cc_func']           = 'ctp_xcorr_norm' # "method" to compute CC :
#       'ctp_xcorr'      => Raw cross-correlation CC = tr1 . tr2*  
#       'ctp_xcorr_norm' => Energy Normalized cross-correlation CC = tr1 . tr2* / ( sqrt( sum(tr1)^2 . sum(tr1)^2 ) )
#       'ctp_coher'      => Spectral normalization,  cross-coherence CC = tr1 . tr2* / ( abs(tr1) *  abs(tr2))
in_['cc_dtype']          = 'float32' # CC precision for storage (float16,float32,float64)
in_['cc_scaling']        = 1000  # scaling factor before lowering precision 
in_['cc_tags']           = 3 # 1 : xcorr only intra-tag data,2 : xcorr only inter-tag data or 3: xcorr all  

############## STACKS
# Finale results are always concatenated in master *.h5 files
# Inside these master files you will find: 
#       - (ALWAYS) the stack over the full period (ref)
#       - short time window correlation table (cc) for monitoring, IF "keep_daily_corr" is switched to True
# Daily (temporary) files can also be kept using "remove_daily_file" option.
in_['hr_stack']          = False  # If True, keep only one corr per day and not smaller time windows (see hr1, hr2)
in_['keep_daily_corr']   = False   # If True, keep daily(/or smaller time window) correlations in the concatenated final file
in_['remove_daily_file'] = False   # If True, remove daily files, and keep only the concatenated keep them for debugging or if you plan to add dates
## Phase weighted stacks?
in_['pws']               = True # apply pws() instead of mean()
in_['pws_timegate']      = 120. # pws params see Schimmel and Paulsen 1997
in_['pws_power']         = 2. # pws params see Schimmel and Paulsen 1997
## svd_wiener2 ? (pws OR svd_wiener2... not both)
in_['svd_wiener2']       = False # apply svd-wiener2 filter before stacking Moreau et al. 2017 (GJI)
in_['svd_wiener2_m']     = 20 # date axis, number of point for wiener window 
in_['svd_wiener2_n']     = 20 # time lag axis, number of point for wiener window
in_['svd_wiener2_nvs']   = None # number of singular value to keep, None will keep only singular values > 10% min value

############## MISC

in_['gzip']              = False # h5 option
in_['pp']                = [] # same as pre-processing recipe ...
#arg_filter = {
#     'type'  : 'bp' # bp, hp or lp
#    ,'f1'    : 1/50.0 # lower corner freq for bp and lp
#    ,'f2'    : 1/5.0 # higher corner freq for bp and hp
#    ,'order' : 2.0 # filter order
#    ,'taper' : 0.01 # between 0 (off) and 1 (hann window)
#}
#in_['pp_args']           = [arg_filter]


############### RUN
xcorr.xcorr_all(in_)

