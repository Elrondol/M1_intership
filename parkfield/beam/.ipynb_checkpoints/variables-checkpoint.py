import numpy as np


in_ = {} # general inputs
#in_['data_dir']            = '/data/projects/faultscan/user/parisnic/m1_internship/parkfield/data_parkfield/' # master dir where to find data_fsHz/  
in_['data_dir']            = '/summer/faultscan/user/parisnic/m1_internship/parkfield/data_parkfield/'
in_['station_file']        = "./stations_rm_nobad_simeon.txt" # station file same as pycorr
in_['tag']                 = 'glob' # same as pycorr ... data_fsHz/daily/tag/year/dailyXXX.h5 
in_['beam_dir']            = 'save_beam/' + in_['tag'] + '/eq_simeon_w3s_40hz_bary_wide' # where do you want to save ?

tt = {} # time inputs
tt['year']                 = 2002 # year, no loop over year to remain simple ... run the code twice if overlap over 2 years
tt['day']                  = list(range(16,17)) # list of julian days
tt['t_0']                  = 15*60*60+29*60+80 # in sec, from first sample of the file (t0 for every day)
tt['t_win']                = 3 #1*60*60 # in sec, entire trace if None    #1 h de window
tt['t_inc']                = 1  #1*60*60 # in sec ? time from one window to the next   #overlap -> ici pas d'overlap
tt['t_end']                = 15*60*60+29*60+90 # in sec ? (tend for every day)

ff = {} # freq inputs
ff['fs']                   = 40.0 # sampling freq
ff['nf']                   = 600 # number of sample in freq within fw # or False if all freq sample

bb = {} # array/beam inputs
bb['U']                    = np.linspace(-1.5,1.5,201)*10**-3 #np.linspace(-1.3,1.3,101)*10**-3 # s/m
bb['compo']                = 'Z' # compo could be Z, E, N, 1, 2, 3 or T, R
bb['beam_type']            = 'bartlett' # classic, bartlett or mvdr
bb['stack_daily']          = False # Stack all fk to form a daily average, save only the daily files if True, save separate files if False 
bb['pre_filt']             = False # True, Flase, pre_filt data ... needed for clipping 
bb['clip']                 = 2 # clip at 2*std of the dataset ... set to False/None if no cliping. Cliping only if pre-filt to True.


pl = {} # plot inputs
pl['fn']                   = [0.95] # list of freq to plot
pl['fw']                   = 3.**(1/2) # float, from fn[i]/fw to fn[i]*fw , if fw=2**(1/2) (an octave),  if fw=2**(1/6) (1/3 of an octave) # 0= full scale,# 1= monochomatic
pl['map']                  = True # plot station map or not ..
pl['clmap']                = 'gist_stern_r' # colormap 
pl['db']                   = False # dB scale ?
pl['norm']                 = False # norm amplitude after freq average for every freq panels
pl['clim']                 = None #[-40,0] # lim colobar
pl['xy_zoom']              = None # [-3,3,-3,3] # xlim ylim
pl['power_fk']             = True # fk = fk**2
pl['deconv']               = {}
pl['deconv']['apply']      = False # apply deconv - Richardson-Lucy starting from arf?
pl['deconv']['iter']       = 5 #number of iterations
pl['deconv']['sigma']      = .5 # apod arf with a Gauss pulse ... to remove edge effect ...
pl['deconv']['clip']       = True # ..cf richardson_lucy clipping
pl['deconv']['filter_epsilon'] = None # see richardson_lucy

pl['target']               = {'dU':None, 'color':'k'} #delta slowness
pl['save_fig']             = in_['beam_dir'] + '/fig/'
pl['arf_only']             = True # plot only arf instead of fk


compo_to_plot   = bb['compo']
use_pre_stacked = True # mettre False si doit utiliser des versions stackées sur la journée !!! 


pl_updated = {} # plot inputs
pl_updated['fw']                   = 3.**(1/2) # float, from fn[i]/fw to fn[i]*fw , if fw=2**(1/2) (an octave),  if fw=2**(1/6) (1/3 of an octave) # 0= full scale,# 1= monochomatic
pl_updated['fn']                   = [0.95]
pl_updated['map']                  = True # plot station map or not ..
pl_updated['clmap']                = 'gist_stern_r' # colormap 
pl_updated['db']                   = False # dB scale ?
pl_updated['norm']                 = False # norm amplitude after freq average for every freq panels
pl_updated['clim']                 = None #[0,0.006] # lim colobar, or None for min-max
pl_updated['xy_zoom']              = None # [-3,3,-3,3] # xlim ylim

pl_updated['deconv']               = {}
pl_updated['deconv']['apply']      = False # apply deconv - Richardson-Lucy starting from arf?
pl_updated['deconv']['iter']       = 5 #number of iterations
pl_updated['deconv']['sigma']      = 0.1 # apod arf with a Gauss pulse ... to remove edge effect ... decrease sigma if unstable deconv
pl_updated['deconv']['clip']       = True # ..cf richardson_lucy clipping
pl_updated['deconv']['filter_epsilon'] = None # see richardson_lucy
pl_updated['target']               = {'dU':None, 'color':'k'} #delta slowness
pl_updated['save_fig']             = in_['beam_dir'] + '/fig/'
pl_updated['arf_only']             = False # plot only arf instead of fk
pl_updated['power_fk']             = False # power...

pl_updated['format']               = 'png'
