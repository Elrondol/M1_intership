################################################
# m20_noise_processing.py 
# Pierre Boue (UGA)
# Laurent Stehly (UGA)
# email: pycorr@univ-grenoble-alpes.fr
# May 2017
################################################

'''
# TO DO

'''

import os
import glob
import h5py
import pickle
import shutil
import random 
import numpy as np 
from numpy.lib.scimath import sqrt as csqrt
import scipy.signal as signal 
from scipy.signal import hilbert
import scipy.fftpack
# OBSPY FOR EQ CATALOG ONLY
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
import m_pycorr.mods.lang as lang
import m_pycorr.mods.dd as dd

try : 
    import ipdb
except :
    pass 


##########################################################################################
##########################################################################################
######         ####          ####       ####   ####          ####   ####         #########
######         ####          ####       ####   ####          ####   ####         #########
######    #########   ####   ####    #   ###   ####    ##########   ####    ##############
######   ##########   ####   ####    #   ###   ####    ##########   ####   ###############
######   ##########   ####   ####    ##   ##   ####       #######   ####   ####    #######
######    #########   ####   ####    ##   ##   ####    ##########   ####    ###  #########
######         ####          ####    ###       ####    ##########   ####         #########
######         ####          ####    ###       ####    ##########   ####         #########
##########################################################################################
##########################################################################################


def apply_recipe(in_dir,inu={}) :
    in_                      = {}
    in_['clean_input']       = False # if True rm input (raw data) day after day ...
    in_['recipe']            = ['_filter','_onebit']
    in_['args']              = [{},{}]
    in_ = lang.merge_options(in_,inu)
    out_dir = in_dir + '_cooked'
    # loop over all methods...
    methods_to_call = []
    for method in in_['recipe']:
        if method in globals():
            methods_to_call.append(globals()[method])
        else:
            dd.dispc('unknown method : ' +  method,'r','b')
            return
    dd.dd(in_)
    _main_loop(in_dir,out_dir,methods_to_call,in_)


# or your own processing ...
def my_processing(in_dir,inu={}) :
    in_                      = {}
    in_['recipe']            = ['_my_processing']
    in_['clean_input']       = False
    in_ = lang.merge_options(in_,inu)
    out_dir = in_dir + '_my_processing'
    methods_to_call = [globals()['_my_processing']]
    _main_loop(in_dir,out_dir,methods_to_call,in_)



#########################################################################################
#########################################################################################
########    ##########            ######            #####            ####################
########    #########     ####     ####     ####     ####    ####     ###################
########    #########    ######    ####    ######    ####    #####    ###################
########    #########    ######    ####    ######    ####    ####     ###################
########    #########    ######    ####    ######    ####           #####################
########    #########    ######    ####    ######    ####    ############################
########          ###     ####     ####     ####     ####    ############################
########          ####            ######            #####    ############################
#########################################################################################
#########################################################################################


def _main_loop(in_dir,out_dir,flist,in_) :
    if 'clean_input' in in_: 
        clean_input = in_['clean_input']
    if '_seg_in' in in_['recipe']:
        if 'cut_evts' in in_['args'][in_['recipe'].index('_seg_in')]:
            clientcat   = Client("IRIS")
    else: clean_input = False

    dd.dispc('inside main loop : working on '+in_dir,'y','b')
   
    mkdir(out_dir)
    year_list = glob.glob(in_dir+'/daily/*/*/')
    year_list.sort()
    random.shuffle(year_list)    
    # loop on year & creating the corresponding output dir 
    for iyear in year_list :

        h5_list = glob.glob(iyear+'*h5')
        h5_list.sort()
        out_year = out_dir + '/daily/'+iyear.split('/')[-3]+'/'+iyear.split('/')[-2]
        mkdir(out_year)
        #copy _metadata and db* file if any :
        _copy_metadata(iyear,out_year)

        ##determining the sampling rate from db.pkl
        #db_file = iyear+'/db.pkl'
        #if glob.glob(db_file):
        #    dbin = pickle.load(open(db_file, "rb" ))
        #    fe = dbin['in_']['pp']['freq']
        #    dd.dispc('fe = ' + str(fe),'y','b')
        #else:
        #    fe = float(in_dir.split('/')[-1].split('_')[1].split('h')[0])

        #loop on daily h5 files : 
        random.shuffle(h5_list)
        for ih5 in h5_list : 
            out_h5      = out_year + '/'+ih5.split('/')[-1]
            out_h5_lock = out_h5+'.lock'
            
            if os.path.isfile(out_h5) | os.path.isfile(out_h5_lock) :
                dd.dispc('  '+out_h5 +' already exist ... continuing','r','r')
                continue 
            #the output file does not exist => we process this file 
            dd.dispc(' working on file '+ih5,'c','b')
            create_lock_file(out_h5_lock)

            # if cut earthquake then find catalog form IRIS
            if '_seg_in' in in_['recipe']:
                if 'cut_evts' in in_['args'][in_['recipe'].index('_seg_in')]:
                    evt_db     = {}
                    eq_dt      = 86400. * 10
                    evt_db['ddate']      = ih5.split('/')[-1].split('.')[0].split('_')[1]
                    evt_db['date_trace'] = UTCDateTime(iyear.split('/')[-2] + evt_db['ddate'])
                    try :
                        mag_lim = in_['args'][in_['recipe'].index('_seg_in')]['mag_lim']
                        evt_db['cat']        = clientcat.get_events(starttime=evt_db['date_trace'] - eq_dt , endtime= evt_db['date_trace'] + 86400.0 , minmagnitude= mag_lim)
                    except:  evt_db['cat'] = [] 
                    #print(evt_db['cat'])

            # open the input & output file, and scann all metadata and waveforms :
            try :
                ff   = h5py.File(ih5,'r')
                fout = h5py.File(out_h5,'w') 
            except : 
                continue

            if '/_metadata' in ff: fout.copy(ff['/_metadata'],'/_metadata')
            #determining the sampling rate from _metadata

            numfct   = 0
            if '/_metadata' in fout:
                numfct_r = len(fout['/_metadata']) - 3
            else : numfct_r = 0

            for fh in flist:
                numfct   += 1
                numfct_r += 1
                fout = add_metadata(fout,'_metadata/' + '_%02d' % float(numfct_r) + fh.__name__,in_['args'][numfct - 1])

            net_list = ff.keys()
            for inet in net_list :
                if inet not in ['_metadata']:
                    fout.create_group(inet)
                    # test d'integrite 1 
                    try :
                        sta_list = ff[inet].keys()
                        len(sta_list)
                    except :
                        continue 
                    for ista in sta_list :
                        group_name='/'+inet+'/'+ista
                        print(group_name)
                        fout.create_group(group_name)
                        # integrity check
                        try :
                            for ich in ff[group_name] :
                                pass
                        except :
                            continue 
                        
                        for ich in ff[group_name]  : 
                        #for ich in ista[1].items() :
                            # get fe init for each trace
                            if '/_metadata/fe' in ff: fe = ff['/_metadata/fe'][()]
                            else : fe = float(in_dir.split('/')[-1].split('_')[1].split('h')[0])
                            dataset_name= '/'+inet+'/'+ista+'/'+ich
                            #print('  '+dataset_name)
                            # integrity check 2 : 
                            try : 
                                trace = ff[dataset_name][:] 
                            except : 
                                continue 
                            #print('    got trace')
                            fout.create_dataset(dataset_name,data = trace)
                            #print('    created dataset')
                            segment      = False
                            method_index = 0
                            for fh in flist:
                                dico_method = {}
                                try: dico_method = in_['args'][method_index]
                                except:pass
                                ####### IN/OUT SUBTRACES PROCESSING
                                if fh.__name__ == '_seg_in':
                                    segment = True # Enter subtreatment
                                    sublen = int(dico_method['len_segment'] * fe)
                                    nb_segment = len(trace) / (float(fe) * float(dico_method['len_segment']))
                                    if nb_segment - int(nb_segment):
                                        dd.dispc(' len(trace) / len_segment should be an integer : modify len_segment...','r','b')
                                        fout.close()
                                        ff.close()
                                        os.remove(out_h5_lock)
                                        return

                                    nb_segment = int(nb_segment)
                                    transients = np.ones(nb_segment)

                                    if dico_method['cut_custom']: ### DO IT ALSO OUTSIDE DEG_IN ????
                                        transients = __detect_transients_and_zeros(trace,float(fe),dico_method,transients)
                                    
                                    if dico_method['cut_evts']:
                                        transients = __detect_earthquakes(trace,float(fe),dico_method,evt_db,transients)

                                    #print(transients)

                                elif fh.__name__ == '_seg_out': segment = False # Get out subtreatment dangerous ...

                                ####### RUN PROCESSING OR CUT
                                else :
                                    if not segment:
                                        trace, nin_ = fh(trace,fe,dico_method)
                                        dico_method = nin_
                                    else:
                                        for il in range(int(nb_segment)):
                                            if transients[il] :
                                                trace[il*sublen:(il+1)*sublen], nin_ = fh(trace[il*sublen:(il+1)*sublen],fe,dico_method)
                                                dico_method = nin_
                                            else:
                                                trace[il*sublen:(il+1)*sublen] = np.zeros(sublen)
                                # update fe ?
                                if 'new_fe' in dico_method :
                                    if  fe is not dico_method['new_fe']:
                                        del fout['/_metadata/fe']
                                        fe = dico_method['new_fe']
                                        fout.create_dataset('/_metadata/fe',data = fe)
                                ####### CUT IF NO PROCESSING AFTER '_seg_in'
                                if flist[-1].__name__ == '_seg_in' and segment:
                                    for il in range(int(nb_segment)):
                                        if not transients[il] :
                                            trace[il*sublen:(il+1)*sublen] = np.zeros(sublen)
                                method_index += 1

                            ####### after all methods ...
                            #ipdb.set_trace()
                            if not max(trace): 
                                del fout[dataset_name]
                            else:
                                if len(fout[dataset_name][:]) is not len(trace):
                                    del fout[dataset_name]
                                    fout.create_dataset(dataset_name,data = trace)
                                else:
                                    fout[dataset_name][:] = trace
                            
            #cleaning
            fout.close()
            ff.close()
            if clean_input:
                dd.dispc(' rm ' + ih5,'r','b')
                os.remove(ih5)
            os.remove(out_h5_lock)





##########################################################################################
##########################################################################################
#########           #####         ####               #####           #####################
#########           ####          ####               ####     ############################
#########     ##########     ##############     #########     ############################
#########     #########     ###############     ##########        ########################
#########        ######     ###############     ##############       #####################
#########     #########      ##############     ##################    ####################
#########     ##########          #########     #################     ####################
#########     ###########         #########     ##########           #####################
##########################################################################################
##########################################################################################


#..............................
#put signal processing function acting on a single trace here .... 
#.................................


'''
if you modify the sampling rate and/or t0 of the data ... don't forget to also modify :
    /_metadata/fe
    /_metadata/t0_UNIX_timestamp
'''


#------------------------------------------
def _my_processing(trace,fe,inu={}):
    # put your code here 
    # ...
    #in_ = {}
    #in_['my_processing'] = 0
    #if 'my_processing' not in inu:
    #    dd.dispc('no my_processing value(s) : using default  ' + str(in_['my_processing']),'r','b')
    #in_ = lang.parse_options(in_,inu)
    # ...
    return trace,inu

#------------------------------------------
def _taper(trace,fe=None,inu={}):
    in_ = {}
    in_['taper'] = 0.2
    in_ = __check_in(in_,inu,'_taper')
    return trace * signal.tukey(len(trace),in_['taper']),in_

#------------------------------------------
def _normenv(trace,fe=None,inu={}):
    return trace / float(abs(hilbert(trace))),inu

#------------------------------------------
def _onebit(trace,fe=None,inu={}):
    return np.sign(trace),inu


#------------------------------------------
def _filter(trace,fe,inu={}):
    in_ = {}
    in_['taper'] = 0.
    in_['order'] = 2
    in_['type']  = 'bp'
    in_['f1']    = float(fe) / 200. 
    in_['f2']    = float(fe) / 2. - 0.05*( float(fe) / 2.)
    in_ = __check_in(in_,inu,'_filter')
    if in_['taper']:
        trace *= signal.tukey(len(trace),in_['taper'])
    dt = 1/float(fe)
    
    if in_['type'] is 'bp':
        corner = np.array([in_['f1'],in_['f2']])*2.*dt
        b,a = signal.butter(in_['order'],corner,'bandpass',analog=False,output='ba')
    if in_['type'] is 'hp':
        corner = np.array(in_['f2'])*2.*dt
        b,a = signal.butter(in_['order'],corner,'highpass',analog=False,output='ba')
    if in_['type'] is 'lp':
        corner = np.array(in_['f1'])*2.*dt
        b,a = signal.butter(in_['order'],corner,'lowpass',analog=False,output='ba')

    if np.ndim(trace)==2 :
        trace = signal.filtfilt(b,a,trace,axis=1) #padtype=None)
    else :
        trace =signal.filtfilt(b,a,trace,axis=0) #padtype=None)

    if np.isnan(trace).any():
        dd.dispc('WARNING : nan detected after filtering, order probably too high','r','b')
    return trace,in_

#------------------------------------------
def _comb_filter(trace,fe,inu={}):
    in_ = {}
    in_['p1'] = [5 ,10,20,40]
    in_['p2'] = [10,20,40,80]
    in_ = __check_in(in_,inu,'_comb_filter')
    nper   = len(in_['p1'])
    trace2 = np.zeros(np.shape(trace))
    for iper in np.arange(0,nper) :
        trace_tmp = __filter(trace,in_['p1'][iper],in_['p2'][iper],1./float(fe),order=2)
        env       = abs(hilbert(trace_tmp))
        trace_tmp = __norm(trace_tmp/env)
        trace2    = trace2 + trace_tmp 
    return trace2 ,in_

#------------------------------------------
def _whitening(trace,fe,inu={}):
    in_ = {}
    in_['f1']  = float(fe) / 400. 
    in_['f2']  = float(fe) / 40. 
    in_['div'] = 10. 
    in_ = __check_in(in_,inu,'_whitening')   
    IndexWhiteMin, IndexWhiteMax, LengthApodisation, BorderLeft, BorderRight, NoramisationWhitening = \
        __prepare_whitening_cos(len(trace), in_['f1'], in_['f2'], fe, in_['div'])
    return __whitening_cos_prepared(trace, IndexWhiteMin, IndexWhiteMax, LengthApodisation, BorderLeft, BorderRight, NoramisationWhitening),in_

#------------------------------------------
def _clipping(trace,fe=None,inu={}):
    in_ = {}
    in_['treshold']  = 4.
    in_['replace']   = 4.
    in_['niter']     = 1.  
    in_ = __check_in(in_,inu,'_clipping')
    if 'niter' in in_:
        if in_['niter']:
            return __n_clipping(trace,fe,in_),in_
        else:
            return __convergent_clipping(trace,fe,in_),in_
    else:
        return __convergent_clipping(trace,fe,in_),in_

#------------------------------------------
def _get_envelope(trace,fe,inu={}):
    # Modified from Piero Poli
    # Description of the processing
    # 1) Filtering [implemented in pycorr]
    # 2) square
    # 3) low pass [implemented in pycorr]
    # 4) resampling
    # 5) sqrt
    # 6) make it real
    # This is now just for resampling at 1hz. It could be good to have other resampling
    in_ = {}
    in_['f1']      = 0.1
    in_['f2']      = 1.
    in_['lowpass'] = 1. 
    in_['new_fe']  = 1. 
    in_            = __check_in(in_,inu,'_get_envelope')    
    # 1)
    dt       = 1/float(in_['new_fe'])
    corner   = np.array([in_['f1'], in_['f2']]) * 2. * dt
    b, a     = signal.butter(2, corner, 'bandpass', analog=False, output='ba')
    trace    = signal.filtfilt(b, a, trace, axis=0)
    # 2)
    trace    = np.power(trace, 2)
    # 3)
    cornerlp = in_['lowpass'] * 2. * dt
    b1, a1   = signal.butter(2, cornerlp, 'lowpass', analog=False, output='ba')
    trace    = signal.filtfilt(b1, a1, trace, axis=0)
    # 4)
    #ipdb.set_trace()
    trace    = signal.decimate(trace, int(fe/float(in_['new_fe'])), ftype="iir",zero_phase=False) ## NEED TO BE MODIFIED
    # 5)
    trace    = csqrt(trace)
    return np.real(trace),in_


#------------------------------------------
def _seg_in():
    #empty method
    return

#------------------------------------------
def _seg_out():
    #empty method
    return


#----experiments---------------------------
def _smoothspectrum(trace,fe=None,inu={}):
    trace = scipy.fftpack.fft(trace, overwrite_x=True)
    w = np.ones(25,'d')
    y = np.convolve(w/float(np.sum(w)),np.absolute(trace),mode='same')
    trace = y * np.exp(1j * np.angle(trace))
    trace = scipy.fftpack.ifft(trace, overwrite_x=True).real
    return trace,inu






#------------------------------------------
# Private functions
#------------------------------------------

def __detect_earthquakes(trace,fe,in_,evt_db,V):
    sublen     = int(fe * in_['len_segment'])

    for evt in evt_db['cat']:
        ev_t    = evt.origins[0].time - 5. * 60.
        
        ######### empirical law...
        treshold_t       = 86400. * (np.exp(evt.magnitudes[0].mag**3 /300.) -1) 
        #########
        
        start_t = 0.
        end_t = 0.
        if ev_t > evt_db['date_trace']:
            start_t = (ev_t - evt_db['date_trace']) * float(fe)
            start_t = int( start_t / sublen)

        if (ev_t + treshold_t) >= evt_db['date_trace'] + 86400.:
            end_t = 86400. * float(fe)
            end_t = int( end_t / sublen)
        elif (ev_t + treshold_t) > evt_db['date_trace'] and (ev_t + treshold_t) < evt_db['date_trace'] + 86400. :
            end_t = (ev_t + treshold_t - evt_db['date_trace']) * float(fe)
            end_t = int( end_t / sublen)
        V[int(start_t ):int(end_t)] = 0.
    return V



def __detect_transients_and_zeros(trace,fe,in_,V):
    sublen     = int(fe * in_['len_segment'])
    nb_segment = int(len(trace) / (fe * in_['len_segment']))
    mean_E     = __meanEnergy(trace,fe,in_)
    for il in range(nb_segment):
        subtrace = trace[il*sublen:(il+1)*sublen] 
        E_trace  = np.sum(np.power(subtrace,2)) / sublen
        std      = np.std(subtrace)
        std1     = np.std(subtrace[0:np.round((subtrace.shape[0])/3)])
        std2     = np.std(subtrace[np.round((subtrace.shape[0])/3):2*np.round((subtrace.shape[0])/3)])
        std3     = np.std(subtrace[0:np.round((subtrace.shape[0]))])
        minstd   = np.amin([std1, std2, std3])
        maxstd   = np.amax([std1, std2, std3])
        condition_E     = E_trace > mean_E * in_['ratio_e']
        condition_STD   = maxstd > in_['ratio_std'] * minstd
        condition_ZEROS = np.size(np.where(np.absolute(subtrace) > in_['zero']),axis=1) < in_['ratio_zero']*len(subtrace)

        if condition_E and condition_STD or condition_ZEROS: V[il] = 0

    return V

#------------------------------------------
def __meanEnergy(trace,fe,in_):
    sublen     = int(fe * in_['len_segment'])
    nb_segment = int(len(trace) / (fe * in_['len_segment']))
    aminiE = np.zeros(nb_segment, dtype = 'float')
    inc = 0
    for i in range(nb_segment):
        SubTrace = trace[i*sublen:(i+1)*sublen]
        if  np.size(np.where(np.absolute(SubTrace) > in_['zero']),axis=1) > in_['ratio_zero']*len(SubTrace):
            aminiE[inc]=np.sum(np.power(SubTrace,2))/len(SubTrace)
            inc = inc + 1
    if np.amax(aminiE) == 0:
        return 0
    else:
        return np.sum(aminiE[np.where(aminiE > 0)]) / inc 

#------------------------------------------
def __whitening_cos_prepared(trace, IndexWhiteMin, IndexWhiteMax, LengthApodisation, BorderLeft, BorderRight, NormalisationWhitening):
    TraceWitened =  np.zeros(len(trace), dtype='complex')
    TraceWitened[IndexWhiteMin:IndexWhiteMax] = scipy.fftpack.fft(trace)[IndexWhiteMin:IndexWhiteMax]
    TraceWitened[IndexWhiteMin:IndexWhiteMax] /= np.absolute(TraceWitened[IndexWhiteMin:IndexWhiteMax])*NormalisationWhitening
    TraceWitened[IndexWhiteMin:IndexWhiteMin+int(LengthApodisation)] *= BorderLeft
    TraceWitened[IndexWhiteMax-int(LengthApodisation):IndexWhiteMax] *= BorderRight
    return ((scipy.fftpack.ifft(TraceWitened, overwrite_x=True)).real)

#------------------------------------------
def __prepare_whitening_cos(LengthTrace, freqMin, freqMax, FreqSampling, DivideFreq):
    dt = (freqMax - freqMin)/DivideFreq
    IndexWhiteMin = int(round(freqMin*LengthTrace/FreqSampling+1))-1
    IndexWhiteMax = int(round(freqMax*LengthTrace/FreqSampling+1))
    LengthApodisation = round(dt/FreqSampling*LengthTrace)
    BorderLeft = np.square(np.sin(np.arange(1,int(LengthApodisation)+1)/LengthApodisation/2*np.pi))
    BorderRight = np.square(np.sin(np.arange(int(LengthApodisation)+1,2*LengthApodisation+1)/LengthApodisation/2*np.pi))
    NoramisationWhitening = np.sqrt(float(IndexWhiteMax-IndexWhiteMin-5*LengthApodisation/4)/LengthTrace/2)
    return IndexWhiteMin, IndexWhiteMax, LengthApodisation, BorderLeft, BorderRight, NoramisationWhitening

#------------------------------------------
def __convergent_clipping(trace,fe=None,in_={'treshold': 5 ,'replace': 5}):
    std1 = np.std(trace[0:int(np.round((trace.shape[0])/3))])
    std2 = np.std(trace[int(np.round((trace.shape[0])/3)):int(2*np.round((trace.shape[0])/3))])
    std3 = np.std(trace[0:int(np.round((trace.shape[0])))])
    minstd = np.amin([std1, std2, std3])       
    arrayReplace = np.ones(len(trace), dtype = 'float')*minstd*in_['replace']
    arrayReplace *= np.sign(trace)
    tr2 = scipy.where(scipy.absolute(trace)>in_['treshold']*minstd, arrayReplace, trace)
    while not np.min(tr2 == trace):#It exists a value different in trace and tr2
        trace = tr2
        arrayReplace = np.ones(len(trace), dtype = 'float')*minstd*in_['replace']
        arrayReplace *= np.sign(trace)
        tr2 = scipy.where(scipy.absolute(trace)>in_['treshold']*minstd, arrayReplace, trace)
    return tr2

#------------------------------------------
def __n_clipping(trace,fe=None,in_={'treshold': 5 ,'replace': 5,'niter': 5}):
    for i in range(in_['niter']):
        arrayReplace = np.ones(len(trace), dtype ='float')*np.std(trace)*in_['replace']
        arrayReplace *= np.sign(trace)
        trace = scipy.where(scipy.absolute(trace)>in_['treshold']*np.std(trace), arrayReplace, trace)
    return trace

#------------------------------------------
def __filter(C,p1,p2,dt,order=2,plot_filter=False) : 
    b,a = signal.butter(order,(2.*dt/p2,2.*dt/p1),'bandpass',analog=False,output='ba')
    if np.ndim(C)==2 :
        C = signal.filtfilt(b,a,C,axis=1) #padtype=None)
    else :
        C=signal.filtfilt(b,a,C,axis=0) #padtype=None)
    return C 

#------------------------------------------
def __norm(C) :
    if np.ndim(C)==2 :
        nwf, nsig=np.shape(C) 

        for isig in np.arange(0,nsig) :
            max_=np.max(np.abs(C[:,isig]))
            if max_ > 0 :
                C[:,isig]=C[:,isig]/max_ 

    elif np.ndim(C)==1 : 
        max_=np.max(np.abs(C))
        if max_ > 0 : 
            C=C/max_ 
    return C

#------------------------------------------
def __onebit(C) : 
    return np.sign(C)




#-----------------------------------
#utility functions  ....
#-------------------------------------
def __check_in(in_,inu,fctname) :
    missing = False
    if not (set(inu.keys()) == set(in_.keys())) :
        dd.dispc(fctname + ': missing/no argument(s) => using default  ','r','b')
        missing = True
    in_ = lang.parse_options(in_,inu)
    if missing : dd.dd(in_)
    return in_

#------------------------------------------
def add_metadata(fout,group_name,inu) : 
    ''' add metadata to the output (=processed) hdf5 file which describe the processing applied 
    it contains the name of the function and the input parameters ''' 
    if group_name in fout:
        dd.dispc('WARNING : ' + group_name + ' already esists : no update in /_metadata','r','b')
    else:
        group=fout.create_group(group_name)
        for ikey in inu : 
            try : 
                dset= np.asarray(inu[ikey])
                group.create_dataset(ikey,data=dset)
            except : 
                continue  
    return fout 


#------------------------------------------
def _copy_metadata(iyear,out_year) :
    # RM EXISTING FILES AND DIR ....
    if os.path.isfile(out_year+'db.mat') : 
        os.remove(out_year+'/db.mat')
    if os.path.isfile(out_year+'db.pkl') :
        os.remove(out_year+'/db.pkl')
    if os.path.isdir(out_year+'/_metadata'):
        shutil.rmtree(out_year+'/_metadata')
    if os.path.isfile(iyear+'db.mat') : 
        shutil.copyfile(iyear+'db.mat',out_year+'/db.mat')
    if os.path.isfile(iyear+'db.pkl') :
        shutil.copyfile(iyear+'db.pkl',out_year+'/db.pkl')
    if os.path.isdir(iyear+'/_metadata'):
        shutil.copytree(iyear+'_metadata',out_year+'/_metadata')

#------------------------------------------
def mkdir(dir_name) : 
    if not os.path.isdir(dir_name) : 
        os.makedirs(dir_name)

#------------------------------------------
def create_lock_file(filename) :
    ff=open(filename,'wb')
    pickle.dump(filename,ff)
    ff.close()
