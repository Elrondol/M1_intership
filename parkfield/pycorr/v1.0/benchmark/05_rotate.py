# This script will run the rotation on all the correlation 
# outputs from ENZ to RTZ 
# results will be stored in C1_04_xcorr_tag_cc_func/RTZ directory
#
########################################################
######################################################## 
########################################################
import m_pycorr.m40_rotate as rotate
in_ = {}

############### Standard Inputs

in_['path']      = 'C1_04_xcorr_cifalps_conti__xcorr_norm/' # input directory to rotate
in_['mode']      = 'all'  #'ref_only' or 'all' (cc + ref)
in_['cc_cmp']    = ['ZZ','RR','RZ','TR','TT'] # outputs component to keep
in_['format']    = 'float16' # output precision

############### RUN
rotate.main_rot(in_)