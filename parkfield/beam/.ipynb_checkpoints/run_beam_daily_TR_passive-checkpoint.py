#!/usr/bin/env python
# coding: utf-8

# # TRICASTIN EXAMPLE
# - loop over days
# - loop over different time windows
# - for a specific f0 or a freq range

# # init packages and define params

# In[1]:

import sys,os,glob,time,copy,warnings,cProfile,shutil,matplotlib
from IPython import display
import h5py as h5
import numpy as np
import scipy.fftpack
import scipy.signal
import scipy.io as io
import skimage.restoration as deconv
from math import cos, sin, radians
import matplotlib.pyplot as plt

import m_beam
import io_beam
import plt_beam
import dd
import lang

from functions import *
from variables import *

import ipdb


ff['freq_opt']             = 'minmax' # or 'minmax' for a range of freq  or 'all' for all positive freq from 0 to fs/2
if ff['freq_opt'] == 'central':
    ff['f0']               = 1/6. #FLOATS ! central freq # 1/3 of an octave around this central freq
    ff['fw']               = 2.**(1/6) # short freq from f0/fw to f0*fw , if fw=2**(1/2) (an octave),  if fw=2**(1/6) (1/3 of an octave) # 0= full scale,# 1= monochomatic
elif ff['freq_opt'] == 'minmax':
    ff['fmin']             = 0.01 # freq min ...
    ff['fmax']             = 2.5 # freq max ...

lon, lat,height            = compute_barycenter(in_['station_file']) #calcule directe les coordonnées du centre avec la fonction
bb['center']               = np.array([lat,lon])
bb['max_radius_selection'] = None # 3000 # in meter from the cntral point or None
bb['target_baz']           = None # if any target direction ... like an EQ back az , for plot purpose

# # Read station list and check the map
# - make selection 
# - compute relative positions
# - plot

# In[2]:


dict_sta = m_beam.manage_station_list(in_,bb)
#dd.dd(dict_sta)
#plt.close('all')
#plt_beam.mmap(111,dict_sta['lon'],dict_sta['lat'],xylabel=True,res=15)


# # MAIN LOOPS
# - turn time information into sample information
# ### LOOP 
#   - OVER DAYS
#     - OVER TIME WINDOWS WITHIN DAYS
#         - read data (=> move that in a function ?)
#         - select short freq samples
#         - compute beam 
#         - stack if necessary 
#     - save

# In[ ]:


N0    = int(tt['t_0']*ff['fs']) #starting sample
Nw    = int(tt['t_win']*ff['fs']) #number of datapoints for t_win
Ni    = int(tt['t_inc']*ff['fs']) # increment between 2 consecutives win               
Ne    = int(tt['t_end']*ff['fs']) # last sample to care about...

sta          = dict_sta['sta']
list_of_keys = dict_sta['list_of_keys']



#loop over days
############################### refine file and dir names

for index_day in tt['day']: # LOOP OVER DAYS

    in_['full_data_dir']       = in_['data_dir']  + "/data_%d.0"%ff['fs'] + "hz/daily/" + in_['tag'] + "/%04d"%tt['year'] + "/"
    in_['name_h5']             = "day_%03d.h5"%index_day
    in_['in_h5']               = in_['full_data_dir']   + "/" + in_['name_h5']

    N0s = (N0+np.array(range((Ne-N0)//Ni))*Ni).tolist() # list of starting samples accross the day
    
    
    for inn in N0s:
        if inn+Nw > Ne:
            N0s = N0s[0:N0s.index(inn)]
            break
        #print(str(inn) + ' - ' + str(inn+Nw+1))

    for index_N0 in N0s: # LOOP OVER TIME WINDOWS OF A GIVEN DAY
        
        print(in_['name_h5'] + ' -  component ' + bb['compo'] + ' - time window [' + str(index_N0/ff['fs']) + ' - ' + str((index_N0+Nw)/ff['fs']) + '] s')
        
        # init data mat
        if bb['compo'] in ['R','T']:
            dataE         = np.zeros([Nw,len(sta)])
            dataN         = np.zeros([Nw,len(sta)])
        else:
            data          = np.zeros([Nw,len(sta)])

        #open h5
        if not os.path.isfile(in_['in_h5']):
            raise ValueError("not such file  " + in_['in_h5'])
        h5f   = h5.File(in_['in_h5'],'r')

        # loop over stations
        for kname in sta.keys():
            locid =  sta[kname]['loc']
            if locid == '':
                locid = '00'
            if bb['compo'] in ['R','T']:
                dataset_nameE= '/'+ sta[kname]['net'] + '/' + sta[kname]['name'] + '.' + locid + '/E'
                dataset_nameN= '/'+ sta[kname]['net'] + '/' + sta[kname]['name'] + '.' + locid + '/N'
                dataset_name = dataset_nameN
            else:
                dataset_name = '/'+ sta[kname]['net'] + '/' + sta[kname]['name'] + '.' + locid + '/' + bb['compo']
            try :
                if (index_N0+Nw)>=h5f[dataset_name].shape[0]:
                    if bb['compo'] in ['R','T']:
                        dataE[0:h5f[dataset_nameE].shape[0],list_of_keys.index(kname)] = h5f[dataset_nameE][index_N0::]
                        dataN[0:h5f[dataset_nameN].shape[0],list_of_keys.index(kname)] = h5f[dataset_nameN][index_N0::]
                    else:
                        data[0:h5f[dataset_name].shape[0],list_of_keys.index(kname)] = h5f[dataset_name][index_N0::]
                else:
                    if bb['compo'] in ['R','T']:
                        dataE[0:Nw,list_of_keys.index(kname)] = h5f[dataset_nameE][index_N0:index_N0+Nw]
                        dataN[0:Nw,list_of_keys.index(kname)] = h5f[dataset_nameN][index_N0:index_N0+Nw]
                    else:
                        data[0:Nw,list_of_keys.index(kname)] = h5f[dataset_name][index_N0:index_N0+Nw]
            except:
                print("    warning: missing data on this stations : " + kname)
                #ipdb.set_trace()
        h5f.close()
        sys.stdout.flush()

        #cn be moved on outside loop
        if bb['pre_filt']:
            b,a = scipy.signal.butter(2,(ff['fmin']/ff['fs']/2.,ff['fmax']/ff['fs']/2.),'bandpass',analog=False,output='ba')
            if bb['compo'] in ['R','T']:
                pad = np.tile(scipy.signal.tukey(np.shape(dataE)[0], alpha=0.05),[np.shape(dataE)[1],1]).transpose()
                # filter
                dataE = scipy.signal.filtfilt(b,a,dataE*pad,axis=0)
                dataN = scipy.signal.filtfilt(b,a,dataN*pad,axis=0)
                # test norm data :
                if bb['clip']:
                    dataE = np.clip(dataE, -bb['clip']*np.std(dataE),+bb['clip']*np.std(dataE))
                    dataN = np.clip(dataN, -bb['clip']*np.std(dataN),+bb['clip']*np.std(dataN))
            else:
                pad = np.tile(scipy.signal.tukey(np.shape(data)[0], alpha=0.05),[np.shape(data)[1],1]).transpose()
                data = scipy.signal.filtfilt(b,a,data*pad,axis=0)
                if bb['clip']:    
                    data = np.clip(data, -bb['clip']*np.std(data),+bb['clip']*np.std(data))
    # PLOT DATA ?
        #plt.close('all')
        #if bb['compo'] in ['R','T']:
        #    ax = plt.subplot(121)
        #    ax.matshow(dataE,aspect='auto')
        #    ax = plt.subplot(122)
        #    ax.matshow(dataN,aspect='auto')
        #else:
        #    ax = plt.subplot()
        #    ax.matshow(data,aspect='auto')
        #fig = plt.gcf()
        #display.clear_output(wait=True)
        #display.display(fig)
        #wait = input("PRESS ENTER TO CONTINUE.")
        #plt.close()
            

        # apply fft and define freq vect
        if bb['compo'] in ['R','T']:
            freq  = scipy.fftpack.fftfreq(dataE.shape[0], 1/ff['fs'])
            dataE = scipy.fftpack.fft(dataE,axis=0, overwrite_x=True)
            dataN = scipy.fftpack.fft(dataN,axis=0, overwrite_x=True)
        else:
            freq  = scipy.fftpack.fftfreq(data.shape[0], 1/ff['fs'])
            data  = scipy.fftpack.fft(data,axis=0, overwrite_x=True)

        if ff['freq_opt'] == 'central':
            ifreq = m_beam.make_short_freq(ff['f0'],freq,ff['fw'],ff['nf'])    
            #print('    f0=' + str(ff['f0']))
            sys.stdout.flush()
        elif ff['freq_opt'] == 'minmax':
            ifreq = m_beam.make_short_freq_range(ff['fmin'],ff['fmax'],freq,ff['nf'])
            #print('    fmin=' + str(ff['fmin']) + ' - fmax=' + str(ff['fmax']))
            sys.stdout.flush()       
        elif ff['freq_opt'] == 'all':
            ifreq = m_beam.make_short_freq_range(0,ff['fs']/2,freq,ff['nf'])
        else:
            print('check freq_opt... default to "all"')
            ifreq = range(len(freq))
        
        # resample freq axis
        if np.isscalar(ifreq):ifreq=[ifreq]
        ff['sfreq'] = freq[ifreq]

        #init fk and arf   
        fk  = np.zeros([len(ifreq),len(bb['U']),len(bb['U'])],dtype=np.complex128)
        
        # compute beam
        if bb['compo'] in ['R','T']:
            fk_tmp,arf = m_beam.beam_H(dataE[ifreq,:],dataN[ifreq,:],ff['sfreq'],bb['U'],dict_sta['pos'],
                                u0=np.matrix([0.,0.])*10**-3,output_compo=bb['compo'],fktype=bb['beam_type'])
        else:
            fk_tmp,arf = m_beam.beam(data[ifreq,:],ff['sfreq'],bb['U'],dict_sta['pos'],u0=np.array([-0.,-0.])*10**-3,fktype=bb['beam_type'])

        if bb['stack_daily']:
            print('stack ABS(FK) ...')
            fk = fk + np.abs(fk_tmp /len(N0s))
        else:      
            save_dic = {
            'in_' : in_,
            'tt'  : tt,
            'ff'  : ff,
            'bb'  : bb,
            'pl'  : pl,
            'fk'  : fk_tmp,
            'arf' : arf,
            'sta' : dict_sta      
            }
            time_string = compute_time_string(index_N0,ff['fs'])
            beam_file = in_['beam_dir'] + '/beam_' + bb['compo'] + '_t0_' + time_string + '_tw_' + "%d"%(tt['t_win']/60) + 'min_' + in_['name_h5']
            io_beam.write_h5(beam_file, save_dic)

    if bb['stack_daily']:
        save_dic = {
        'in_' : in_,
        'tt'  : tt,
        'ff'  : ff,
        'bb'  : bb,
        'pl'  : pl,
        'fk'  : fk,
        'arf' : arf,
        'sta' : dict_sta          
        }
        beam_file = in_['beam_dir'] + '/beam_' +  bb['compo'] + '_' + in_['name_h5']
        io_beam.write_h5(beam_file, save_dic)

