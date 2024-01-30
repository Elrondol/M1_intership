# This script will pre-processes all data in your archive
#using a custom recipe given in in_['recipe']
# /data_*freq*Hz/ will be copied/replaced by /data_*freq*Hz_cooked/

# This script can be run multiple times
# A natural parallelization will be done a 
# daily basis using temproray *.lock or *.cplt files

import m_pycorr.m20_noise_processing as noise_processing

###############
# MAIN ARGS 
###############

data_path                = './data_1.0hz' # archive to process

in_ = {}
in_['clean_input']       =  False # if True rm input ('data_path') day after day ...


# PROCESSING RECIPE : IT WILL APPLY IN THE SPECIFIED ORDER
# available methods for recipe :
#        _normenv       : time domain norm by abs(hilbert(trace))
#        _onebit        : time domain one bit norm
#        _taper         : tukey taper
#        _filter        : butter filtfilt bp, hp or lp
#        _clipping      : time domain clipping with a*std treshold
#        _whitening     : frequency domain norm with taper
#        _get_envelope  : filter and sqrt and resampling for tremor detection (POLI) 
#        _comb_filter   : time domaine "broadband" one bit norm
#        _seg_in        : enter (sub)segmentation with possible muting
#        _seg_out       : exit (sub)segmentation
#        _yourown       : whatever you add in m01_noise_processing module, see m20_noise_processing 

in_['recipe']            = ['_seg_in','_whitening'] # the list of function/method you want to apply



###############################
# SUB ARGS DEPENDING ON RECIPE
###############################


# arguments for each method are given as a list (in_['args']) of dictionaries
# order in the list should be the same as the order in the recipe ! 

in_['args'] = []
#in_['args'] = [{arg first method},{arg second method}, ....]



#### SOME DEFAULTS ARGUMENTS 

arg_filter = {
     'type'  : 'bp' # bp, hp or lp
    ,'f1'    : 1/200.0 # lower corner freq for bp and lp
    ,'f2'    : 1/1.0 # higher corner freq for bp and hp
    ,'order' : 4.0 # filter order
    ,'taper' : 0.01 # between 0 (off) and 1 (hann window)
}

arg_clipping = {
     'treshold' : 10. #3.5 # treshold * std
    ,'replace'  : 10. #3.5 # replace * std
    ,'niter'    : 1 # if 0 then it converges...
}

arg_seg_in = {
     'len_segment' : 3600*4. # in sec
    ,'cut_custom'  : False # rm detected transients and Zeros
    ,'cut_evts'    : False # rm detected event using a (online) catalog
    
    # if cut_custom
    #,'ratio_e'     : 1.5 # cut if E(subtrace) > ratioE * mean(E(all non-zero subtrace))   ===> cut strong events
    #,'ratio_std'   : 1.2 # AND (with E test) maxstd > ratioSTD*minstd . 3 std per subtrace. ===> cut strong events and not daily variations !
    #,'ratio_zero'  : 0.95 # cut subtrace if less than ratioZero abs(values) are > Zeros
    #,'zero'        : 10**-12 # not exactly 0 ...

    # if cut_evts
    ,'mag_lim'     : 5.0 # min magnitude to care about ...
}

arg_whitening = {
     'f1'  : 1/100.0 # lower corner freq
    ,'f2'  : 1/5.0 # lower corner freq
    ,'div' : 10.0 # lenght of (cos)taper = bandwidth / div
}

arg_taper = {
     'taper' : 0.1 # tukey win :  0 (off) to 1 (hann window)
}

arg_comb_filter = {
     'p1': [2.5 ,5 ,10,20,40,80] # period min vector...
    ,'p2': [5, 10,20,40,80,160] # period max vector...
}

arg_get_envelope = {
     'f1': 0.01  # min frequency ...
    ,'f2': 0.1   # max frequency ...
    ,'new_fe' : 1. # resampling frequency
    ,'lowpass' : 0.5 # lowpass frequency ??
}


arg_others = {'no_argument' : []}

#### CREATING THE LIST ACCORDING TO RECIPE
for method in in_['recipe']:
    if 'arg' + method in vars():
        in_['args'].append(vars()['arg' + method])
    else:
        in_['args'].append(vars()['arg_others'])



###############################
# RUN
###############################
noise_processing.apply_recipe(data_path,in_)




