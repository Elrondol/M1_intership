################################################
# m31_xcorr_ev.py 
# Pierre Boue (UGA)
# email: pycorr@univ-grenoble-alpes.fr
# June 2019
################################################



import pickle         
import numpy as np 
import scipy.fftpack
import scipy.io as io
import scipy.signal as signal
import random
import h5py
import glob
import os
import copy
import inspect,sys
import time 
from obspy.core import UTCDateTime

import m_pycorr.mods.lang as lang
import m_pycorr.mods.dd as dd
import m_pycorr.m10_get_data as get_data
import m_pycorr.m20_noise_processing as noise_processing
from m_pycorr.m30_xcorr import *

try : import ipdb 
except : pass 

"""
"""


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
    
def xcorr_all_ev(inu) :
    # load each db file and count the number of station : 
    db={}
    inu['date']=[]
    for iset in inu['path']  : 
        ff = open(iset+'/db.pkl','rb')
        db[iset] = pickle.load(ff)
        ff.close()
        for ev in db[iset]['ev']:
            inu['date'].append(UTCDateTime(db[iset]['ev'][ev]['date']).timestamp)
    ex = xcorr_ev(inu,db,db[[*db][0]]['in_']['pp']['freq'])
    main_loop_ev(ex)
    
def xcorr_from_station_and_event_list(inu,station_file,event_file,fe,cut_len) :
    ''' quickly written. To be done properly later'''
    db={}
    ff=open(station_file,"r")
    lines=ff.readlines()
    ff_ev=open(event_file,"r")
    lines_ev=ff_ev.readlines()     
    for iset in inu['path']:
        db[iset]=dict()
        db[iset]['in_']={'tag' : 'set_0','pp' :{'cut_len' :cut_len}}
        db[iset]['sta']=dict()
        db[iset]['ev']=dict()   
        for iline in lines:
            cline=iline.split(' ')
            if cline[3]=='--':
                kname=cline[1]+'_'+cline[2]+'_00'
            else:
                kname=cline[1]+'_'+cline[2]+'_'+cline[3]
            db[iset]['sta'][kname]={}
            db[iset]['sta'][kname]['name'] = cline[2]
            db[iset]['sta'][kname]['lat']  = float(cline[4])
            db[iset]['sta'][kname]['lon']  = float(cline[5])
            db[iset]['sta'][kname]['elev'] = float(cline[6])
            db[iset]['sta'][kname]['depth']= float(cline[7])
            db[iset]['sta'][kname]['dc']   = cline[0]
            db[iset]['sta'][kname]['net']  = cline[1]
            db[iset]['sta'][kname]['loc']  = cline[3] #string !
            db[iset]['sta'][kname]['kname']= kname
        for iline in lines_ev:
            cline=iline.split(' ')
            kname=cline[0] + '_' + cline[-1].split('\n')[0] + '_' + cline[-4]
            db[iset]['ev'][kname]={}
            db[iset]['ev'][kname]['date']    = cline[0]
            db[iset]['ev'][kname]['ev_id']   = kname.replace(':','-')
            db[iset]['ev'][kname]['lat']     = float(cline[3])
            db[iset]['ev'][kname]['lon']     = float(cline[6])
            db[iset]['ev'][kname]['depth']   = float(cline[9])
            db[iset]['ev'][kname]['mag']     = float(cline[-4])
            db[iset]['ev'][kname]['mag_type']= cline[-1].split('\n')[0] 
    ff.close()
    ff_ev.close()
    ex=xcorr_ev(inu,db,fe)
    main_loop_ev(ex)


def xcorr_ev(inu,db,fe) : 
    '''ex contain the following dict : 
    md_c     : metadata common to all cc's : cc'stime vector, all dates, tau, ... 
    set      : dict of all cc set : noise data path, outdir, station pair metadata
    in_      : input parameters of the correlation code
    out_dir  : output directory where the correlation are stored  : C1_xxx_
    '''
    in_   = {}
    in_['path']              = ['../data_5hz/daily/set1','../data_5hz/daily/set2']
    in_['path_out']          = './'
    in_['date']              = [1504846160.0,1279925473.0]
    in_['start_time']        = 1000 # start correlating n sec afetr source time 
    in_['time_win']          = 1000 # time window to correlate in sec
    in_['time_overlap']      = 0 # amount of overlap between windows (s) 
    in_['cc_maxlag']         = 600           # [s] 
    in_['cc_cmp']            = ['ZZ']        # list des channels a correler. 
    in_['cc_func']           = 'ctp_xcorr_norm'
    in_['cc_dtype']          = 'float16'     # 
    in_['cc_scaling']        = 1e12          #multiply CC by a constant (to store them in 1e12, except if xcorr_norm)
    in_['cc_tags']           = 3             # 1 : xcorr only intra-tag data,2 : xcorr only inter-tag data or 3: xcorr all  
    in_['file_size']         = 0.05          # maximum size of each h5 file ! 
    in_['gzip']              = False         # compress correlations ?
    in_['keep_event_corr']   = True    # remove or not daily corr (i.e keep only the stack)
    in_['event_stack']       = False         # if more than one : stack each segments or not ?
    in_['remove_event_file'] = False         # rmove all event files
    in_['pws']               = False  # default = linear stack
    in_['pws_timegate']      = 25.
    in_['pws_power']         = 2.
    in_['svd_wiener2']       = False # apply svd-wiener2 filter before stacking Moreau et al. 2017 (GJI)
    in_['svd_wiener2_m']     = 20 # date axis, number of point for wiener window 
    in_['svd_wiener2_n']     = 20 # time lag axis, number of point for wiener window
    in_['svd_wiener2_nvs']   = None # number of singular value to keep, None will keep only singular values > 10% min value    
    in_['remove_event_file'] = True  # rm or not daily files (keep them for debugging or if you plan to add dates
    in_['pp']                = []  #['_comb_filter']# list of pre-processing to be applied 
    in_['pp_args']           = []  #[{'p1' :[1,5,10,20,40], 'p2' : [5,10,20,40,80]} ]
    in_['tag']               = 'test'

    in_ = lang.parse_options(in_,inu)
    dd.dd(in_)
    #get script that called the code name : 
    script_name = inspect.stack()[2]
    caller = script_name[1][0:-3].split('/')[-1]
    #
    ex={}
    ex['set']={}
    ex['out_dir']= in_['path_out']+'/C1_'+caller+'_'+ in_['tag'] #+'_'+in_['cc_func'][4:-1]+in_['cc_func'][-1]  #to be changed later 
    for ipp in in_['pp'] :
        ex['out_dir']+='_'+ipp
    ex['out_dir']+='__'+in_['cc_func'][4:-1]+in_['cc_func'][-1]  
    if in_['gzip'] : 
        ex['out_dir']+='_gzip'
    ex['in_']=copy.deepcopy(in_)

    npath_per_file=ex_determine_npath_per_h5_file_ev(in_,fe) 

    #list all station pair and their metadata as list => [md]: 
    for ipath1, kpath1 in enumerate(in_['path']): 
        for ipath2, kpath2 in enumerate(in_['path']) : 
            if ipath2 < ipath1: continue 
            if in_['cc_tags']==2 and ipath2==ipath1:continue
            if in_['cc_tags']==1 and ipath2!=ipath1:continue
            sta_pair=[]
            md={}
            md['lat']   = []
            md['lon']   = []
            md['elev']  = []
            md['depth'] = []
            md['id']=[]
            for ista1, ksta1 in enumerate(db[kpath1]['sta']) :
                for ista2, ksta2 in enumerate(db[kpath2]['sta']) :
                    if (ipath1 == ipath2) and (ista2 < ista1) : continue
                    md['lat'].append([db[kpath1]['sta'][ksta1]['lat'],db[kpath2]['sta'][ksta2]['lat']])
                    md['lon'].append([db[kpath1]['sta'][ksta1]['lon'],db[kpath2]['sta'][ksta2]['lon']])
                    md['elev'].append([db[kpath1]['sta'][ksta1]['elev'],db[kpath2]['sta'][ksta2]['elev']])
                    md['depth'].append([db[kpath1]['sta'][ksta1]['depth'],db[kpath2]['sta'][ksta2]['depth']])
                    id0 = db[kpath1]['sta'][ksta1]['kname'].replace('_','.')
                    id1 = db[kpath2]['sta'][ksta2]['kname'].replace('_','.')
                    md['id'].append([id0,id1])
                    sta_pair.append([ista1, ista2])
            #now split station pair per group of npath per file and choose the name of the ouput file :
            npath_in_this_subset=0
            nsubset=-1
            for ipair in range(0,len(sta_pair)) : 
                if npath_in_this_subset==0 :
                    nsubset=nsubset+1 
                    iset1_str=db[kpath1]['in_']['tag']
                    iset2_str=db[kpath2]['in_']['tag']
                    nsubset_str=str(nsubset).zfill(3)
                    set_name=ex['out_dir']+'/xcorr_'+iset1_str+'_'+iset2_str+'_'+nsubset_str
                    ex['set'][set_name]={}
                    ex['set'][set_name]['path1']=kpath1
                    ex['set'][set_name]['path2']=kpath2
                    ex['set'][set_name]['out_dir'] =set_name 
                    ex['set'][set_name]['out_file_pid']=set_name+'.'+str(os.getpid())+'.h5'
                    ex['set'][set_name]['out_file']=set_name+'.h5'
                    ex['set'][set_name]['out_file_lock']=set_name+'.*.lock'
                    ex['set'][set_name]['out_file_lock_pid']=set_name+'.'+str(os.getpid())+'.lock'
                    ex['set'][set_name]['md']={}
                    ex['set'][set_name]['md']['lat'] =[]
                    ex['set'][set_name]['md']['lon'] =[]
                    ex['set'][set_name]['md']['id']  =[]
                    ex['set'][set_name]['md']['elev']=[]
                    ex['set'][set_name]['md']['depth']=[]

                ex['set'][set_name]['md']['lat'].append(md['lat'][ipair])
                ex['set'][set_name]['md']['lon'].append(md['lon'][ipair])
                ex['set'][set_name]['md']['elev'].append(md['elev'][ipair])
                ex['set'][set_name]['md']['depth'].append(md['depth'][ipair])
                ex['set'][set_name]['md']['id'].append(md['id'][ipair])
                npath_in_this_subset=len(ex['set'][set_name]['md']['lat'])
                if npath_in_this_subset >= npath_per_file :
                    npath_in_this_subset=0 
    #display the number of h5 file we will have :
    dd.dispc('  => we will have '+str(len(ex['set']))+' h5 files','y','b')

    #convert each set metadata in numpy array 
    for kset in ex['set'] : 
        ex['set'][kset]['md']['lat']   = np.array(ex['set'][kset]['md']['lat'],dtype='float64')
        ex['set'][kset]['md']['lon']   = np.array(ex['set'][kset]['md']['lon'],dtype='float64')
        ex['set'][kset]['md']['id']    = np.array(ex['set'][kset]['md']['id'],dtype='S20')
        ex['set'][kset]['md']['id']    = ex['set'][kset]['md']['id'][:]
        ex['set'][kset]['md']['elev']  = np.array(ex['set'][kset]['md']['elev'],dtype='float64')
        ex['set'][kset]['md']['depth'] = np.array(ex['set'][kset]['md']['depth'],dtype='float64')
    #for debugging purpose : 
    nsta=0
    for key, value in db.items() : nsta=nsta+len(value['sta'])
    ncpl=round((nsta*(nsta-1))/2)+nsta
    
    ncpl2=0
    for ikey, ivalue in ex['set'].items() : ncpl2=ncpl2+len(ivalue['md']['lat'])
    dd.dispc('  got '+str(ncpl2)+' paths out of '+str(ncpl),'y','n')
    # end of debugging :
    ex['md_c']=ex_get_md_c_ev(db,in_,fe)
    return ex


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

#------------------------------------------------------------------------------------------
def main_loop_ev(ex) : 
    ''' loop on each set of correlation. First put one core per set of correlations 
    in a second step distribute several cores per set of ccs 
    => there are two layers of multiprocessing'''
    print('----------------------')
    print('now in main loop')
    #1st loop : correlate the next set where no one is working (i.e no outdir and its not finished)
    for kset in ex['set'].values():
        if os.path.isdir(kset['out_dir']) : continue 
        if os.path.isfile(kset['out_file']) or os.path.isfile(kset['out_file_lock']) : continue 
        mkdir(kset['out_dir'])
        dd.dispc('  1st loop : working on '+kset['out_dir'],'c','b')
        correlate_this_set_ev(ex['in_'],ex['md_c'],kset)
        # if not all daily correlations have been made : continue
        if ml_have_all_days_been_done_ev(ex['in_'],kset,ex['md_c']) == False : continue 
        #if someone_stacking : continue 
        if len(glob.glob(kset['out_file'][0:-2]+'*h5'))>0 or len(glob.glob(kset['out_file_lock']))>0 : continue 
        ml_stack_and_write_this_set_in_hdf5_ev(ex['in_'],ex['md_c'],kset) 


    # reloop in a random order : now allow several process to work in the same set :
    set_=ex['set'].values()
    set_=list(set_)
    random.shuffle(set_) 
    for kset in set_ :
        if os.path.isfile(kset['out_file']) or os.path.isfile(kset['out_file_lock']) : continue 
        dd.dispc('  2nd loop : working on '+kset['out_dir'],'c','b')
        mkdir(kset['out_dir']) # not useful 
        correlate_this_set_ev(ex['in_'],ex['md_c'],kset) 
        # if not all daily correlations have been made : continue
        if ml_have_all_days_been_done_ev(ex['in_'],kset,ex['md_c']) == False : continue 
        # if someone_stacking : continue 
        if len(glob.glob(kset['out_file'][0:-2]+'*h5')) or len(glob.glob(kset['out_file_lock']))>0 : continue 
        ml_stack_and_write_this_set_in_hdf5_ev(ex['in_'],ex['md_c'],kset) 


#----------------------------------------------------------------------------------------------
def correlate_this_set_ev(in_,md_c,kset) : 
    ''' compute a set of correlations i.e correlations stored in a XCORR/xcorr_set1_set1_000/ dir'''
    mkdir(kset['out_dir'])
    #some shortcuts : 
    maxlag=int(in_['cc_maxlag']/md_c['tau'])
    dtype=in_['cc_dtype']
    cc_func=in_['cc_func']

    # determine .mat file name containing each daily correlation :

    list_ev = []
    for  ilist1 in md_c[kset['path1']]['id_ev']:
        if ilist1 in md_c[kset['path2']]['id_ev']:
            list_ev.append(ilist1)
    
    ev_list=cts_get_filename_list_ev(in_['date'],kset['out_dir'],list_ev)
    random.shuffle(ev_list)
    #loop on each date : 
    for kdate in ev_list :
        md_c_ev        = {}
        md_c_ev        = md_c[kset['path1']][kdate['ev']] # should be both in path1 and path2...
        md_c_ev['tau'] = md_c['tau']
        md_c_ev['t']   = md_c['t']
        I1=np.array(md_c_ev['I1'],dtype='int')
        I2=np.array(md_c_ev['I2'],dtype='int') 
        idate= in_['date'].index(kdate['date'])  # => il faut retrouver l'indice du jour  
        #mondain stuff 1st 
        if cts_is_this_date_already_done(kdate) :
            cts_print_already_done_message(kdate)
            continue 
        cts_print_welcome_message(kdate)
        create_lock_file(kdate['out_lock'])
        #initialize the h5 file containing the cc for this days in any case :
        h5_daily=cts_init_h5_ev_file(kdate,in_,kset['md'],md_c_ev,idate)

        #open h5 files containing noise data, w/o loading it into memory 
        h5a, h5b, success =  cts_open_h5_file_ev(kdate,kset) 
        #if there is no data for this day : close the h5 daily file 

        if success == False : 
            dd.dispc('     : could not open the data file (does it exists?)','r','r')
            h5_daily.close()
            rename(kdate['out_pid'],kdate['out'])
            remove(kdate['out_lock'])
            continue 
        # compute all correlations by pair of stations :
        dd.dispc('                         : '+_cts_get_current_hr_as_str()+' : cross-correlating','c','n')

        h5a_name      = h5a.filename
        h5b_name      = h5b.filename
        h5_daily_name = h5_daily.filename 
        h5a.close()
        h5b.close()
        h5_daily.close()

        for icmp, kcmp in enumerate(in_['cc_cmp']):
            sys.stderr.flush()
            sys.stdout.flush()
            for icpl, kcpl in enumerate(kset['md']['id']): 
                key0=kcpl[0].decode('utf8')+'.'+kcmp[0]
                key1=kcpl[1].decode('utf8')+'.'+kcmp[1]
                h5a      = h5py.File(h5a_name,'r')
                h5b      = h5py.File(h5b_name,'r')
                h5_daily = h5py.File(h5_daily_name,'a')
                try :
                    trace0 = cts_read_data(h5a,kset['md']['id'][icpl:icpl+1],[kcmp],0)
                    trace1 = cts_read_data(h5b,kset['md']['id'][icpl:icpl+1],[kcmp],1)
                    if in_['pp']:
                        dd.dispc('                         : '+_cts_get_current_hr_as_str()+' : pre processing 1st set','c','n')
                        trace0=cts_preprocess_data(trace0,in_['pp'],in_['pp_args'],md_c['tau'])
                        dd.dispc('                         : '+_cts_get_current_hr_as_str()+' : pre processing 2nd set','c','n')
                        trace1=cts_preprocess_data(trace1,in_['pp'],in_['pp_args'],md_c['tau'])
                    cc_hr, ncorr=correlate_this_path(trace0[key0],trace1[key1],maxlag,I1,I2,cc_func,dtype)
                    #multiply the correlation by a constant 
                    if in_['cc_func'] != 'ctp_xcorr_norm' : 
                        cc_hr = cc_hr*in_['cc_scaling']
                    if in_['event_stack']     :
                        #on suppose ici que chaque segement correle a la meme longueur :
                        h5_daily['cc_nstack'][icmp,icpl,0]=sum(ncorr) #(I2[0]-I1[0])*md_c['tau']*sum(ncorr)/86400.0
                        dset_name=h5_get_station_pair_tree_path(kcpl,kcmp)
                        if in_['pws']:
                            h5_create_dataset(h5_daily,'/cc'+dset_name,pws(cc_hr,in_['pws_timegate'],in_['pws_power'],1/float(md_c['tau']),in_['cc_dtype']),in_['gzip'],in_['cc_dtype'])
                        elif in_['svd_wiener2']:
                            h5_create_dataset(h5_daily,'/cc'+dset_name,svd_wiener2(cc_hr,in_['svd_wiener2_m'],in_['svd_wiener2_n'],in_['svd_wiener2_nvs'],in_['cc_dtype']),in_['gzip'],in_['cc_dtype'])           
                        else :
                            h5_create_dataset(h5_daily,'/cc'+dset_name,cc_hr.sum(axis=0),in_['gzip'],in_['cc_dtype'])
                    if not(in_['event_stack']):
                        h5_daily['cc_nstack'][icmp,icpl,:]=ncorr #ncorr*(I2-I1)*md_c['tau']/86400.
                        dset_name=h5_get_station_pair_tree_path(kcpl,kcmp)
                        h5_create_dataset(h5_daily,'/cc'+dset_name,cc_hr,in_['gzip'],in_['cc_dtype'])
                        # end of this day : close all opened h5 files :            
                except:
                    continue
                h5a.close()
                h5b.close()
                h5_daily.close()  

        # end of this day : close all opened h5 files :
        #h5a.close()
        #h5b.close()
        #h5_daily.close()
        rename(kdate['out_pid'],kdate['out'])
        remove(kdate['out_lock'])
        

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


#------------------------------------------------------------------------------------
#                                 scheduler subfunctions 
#------------------------------------------------------------------------------------
def ex_get_md_c_ev(db,in_,fe) : 
    #ipdb.set_trace()
    md_c={}
    md_c['version']= 0.9
    md_c['tau']  =1./fe #db[[*db][0]]['in_']['pp']['freq']    #    
    md_c['t'] = np.linspace(-in_['cc_maxlag'],in_['cc_maxlag'],2*in_['cc_maxlag']*1/md_c['tau']+1)
    #add the list of components from in_ : so that it cab be changed later if we rotate the CC : 
    md_c['date1']=[]
    md_c['date2']=[]
    for iset in in_['path']:
        md_c[iset] = {}
        md_c[iset]['tau']   = md_c['tau']
        md_c[iset]['t']     = md_c['t']
        md_c[iset]['id_ev'] = []
        for ev in db[iset]['ev']:
            md_c[iset][ev] = {} 
            md_c[iset]['id_ev'].append(ev)
            #info sur les donnes de bruit
            time={}
            time['fe']   = 1./md_c['tau'] 
            time['t1']   = in_['start_time'] # t1 en [s] 

            if db[iset]['in_']['ev']['time_after']:
                time['t2']   = db[iset]['in_']['ev']['time_after']
            else:
                time['t2'] = get_data.length_events(db[iset]['ev'][ev]['mag'])

            time['npts'] =int(time['fe']*(time['t2']))          #
            time['time']= np.arange(time['t1'],time['t2'],1.0/time['fe'])   # in second w.r to midnight
                    
            hr1 = np.arange(in_['start_time'],time['t2']-in_['time_win'],in_['time_win']-in_['time_overlap'])         
            hr2 = hr1+in_['time_win']

            #hr1 = np.arange(in_['start_time'],time['t2']-in_['time_win'],in_['time_win'])/3600.
            #hr2 = np.arange(in_['start_time']+in_['time_win'],time['t2'],in_['time_win'])/3600.

            ds1=np.array(hr1) # difference de temps [s]
            ds2=np.array(hr2) # entre chaque fen et le t1 des traces

            #keep the indice of each hourly slice :
            md_c[iset][ev]['I1']=np.round(ds1*time['fe']).astype('int')                 # idem ms en nb de points
            md_c[iset][ev]['I2']=np.round(ds2*time['fe']).astype('int')
            if md_c[iset][ev]['I2'][-1] > time['npts'] : 
                dd.dispc('  ex_get_h5_time_vector : will try to correlate time that do not exist','r','b')

            #construct date vector : 
            md_c[iset][ev]['date1']=[]
            md_c[iset][ev]['date2']=[]
            idate = db[iset]['ev'][ev]['date']
            if in_['event_stack'] :
                    md_c[iset][ev]['date1'].append((UTCDateTime(idate)+time['t1']).timestamp)
                    md_c[iset][ev]['date2'].append((UTCDateTime(idate)+time['t2']).timestamp)
            if not in_['event_stack'] :
                for ihr in range(0,len(hr1)) :
                    md_c[iset][ev]['date1'].append((UTCDateTime(idate)+(hr1[ihr])).timestamp)
                    md_c[iset][ev]['date2'].append((UTCDateTime(idate)+(hr2[ihr])).timestamp)
            md_c['date1'].extend(k for k in md_c[iset][ev]['date1'][:])
            md_c['date2'].extend(k for k in md_c[iset][ev]['date2'][:])
    md_c['date1'].sort()
    md_c['date2'].sort()
    return md_c

def ex_determine_npath_per_h5_file_ev(in_,fe) :
    size_single_path = (in_['cc_maxlag']*2.0+1.0) * fe * len(in_['cc_cmp'])
    if in_['cc_dtype']=='float64'  : size_single_path = size_single_path * 8.0
    elif in_['cc_dtype']=='float32'  : size_single_path = size_single_path * 4.0
    elif in_['cc_dtype']=='float16'  : size_single_path = size_single_path * 2.0
    else : 
        dd.dispc(' cc_dtype different from float16/32/64','y','r')
    ndate=1
    if in_['keep_event_corr'] :
        ndate = ndate *len(in_['date'])
        dd.dispc('  (considering only ONE correlation per event)','y','b')
    size_single_path=size_single_path*ndate/1024./1024./1024. #in Go
    npath_per_file=int(in_['file_size']/size_single_path)
    dd.dispc('  we have '+str(ndate)+' events to save and '+str(len(in_['cc_cmp']))+' cmp','y','b')
    dd.dispc('  => we will put at most '+str(npath_per_file)+' paths per h5 file','y','b')
    return npath_per_file 



#------------------------------------------------------------------------------
#                            main loop subfunctions 
#_-----------------------------------------------------------------------------
def ml_have_all_days_been_done_ev(in_,kset,md_c) : 
    list_ev = []
    for  ilist1 in md_c[kset['path1']]['id_ev']:
        if ilist1 in md_c[kset['path2']]['id_ev']:
            list_ev.append(ilist1)
    ev_list=cts_get_filename_list_ev(in_['date'],kset['out_dir'],list_ev)
    for iev in ev_list : 
        if os.path.isfile(iev['out'])== False :
            return False 
    return True             


def ml_stack_and_write_this_set_in_hdf5_ev(in_,md_c,kset) :
    ''' subtilite ici : qd on efface un tableau d'un fichier h5, le fichier ne change pas de taille 
    => on concatene les correlations dans fichier hdf5 temporaire (daily_matrix_file) 
       et on le copy dans le fichier h5 final que si in_['keep_only_ref']=True. 
    Pour eviter une improbable collision entre deux process qui stackeraient le meme set (si par hasard 
    deux processes creent en meme temps chacun un fichiers .lock) on ecrit d'abord le stack dans un 
    fichier *.pid.h5 avant de le renommer avec son nom final *.h5
    '''
    create_lock_file(kset['out_file_lock_pid'])
    dd.dispc('  stacking '+kset['out_dir'],'y','b')
    daily_matrix_file = kset['out_file_pid'][0:-3]+'_cc.h5'
    t0=time.time()

    #attempt to create the h5 file containing all correlations (may crash => h5 bug :/)
    try : 
        ff   =h5py.File(kset['out_file_pid'],'a')
    except :    
        remove(kset['out_file_lock_pid'])
        remove(kset['out_file_pid']) 
        dd.dispc('  : failed to create '+kset['out_file_pid'],'r','b')
        return 

    #attempt to create the temporary h5 file containing the daily correlation matrix 
    try :
        ff_cc=h5py.File(daily_matrix_file,'a')
    except : 
        ff.close()
        remove(kset['out_file_lock_pid'])
        remove(kset['out_file_pid'])
        remove(daily_matrix_file)
        dd.dispc('  : failed to create '+daily_matrix_file,'r','b')
        return 


    #first convert the date into a matlab readable format : 
    md_c['date1_ordinal']=[]
    md_c['date2_ordinal']=[]
    for idate in range(0, len(md_c['date1'])) : 
        cdate=UTCDateTime(md_c['date1'][idate]) 
        mdate1=cdate.toordinal()+(cdate.hour*3600+cdate.minute*60+cdate.second)/86400.
        cdate=UTCDateTime(md_c['date2'][idate]) 
        mdate2=cdate.toordinal()+(cdate.hour*3600+cdate.minute*60+cdate.second)/86400.
        md_c['date1_ordinal'].append(mdate1)
        md_c['date2_ordinal'].append(mdate2)

    #copy metadata into the hdf5 file : 
    h5_copy_dict_to_group_ev({'in_':in_, 'md':kset['md'], 'md_c' : md_c},ff)

    #special case : copying pp metadata : 
    for ipp in range(0,len(in_['pp'])) :
        #if ipp==0 : 
        #    ff.create_dataset('/in_/pp',data=in_['pp'])
        if len(in_['pp_args']) < ipp : continue 
        for ikey, ival in in_['pp_args'][ipp].items(): 
            ff.create_dataset('/in_/pp_'+in_['pp'][ipp]+'_'+ikey,data=ival)

    #recheck that no other process is doing the same stack : 
    if len(glob.glob(kset['out_file_lock']))>1 :
        remove(kset['out_file_lock_pid'])
        remove(kset['out_file_pid'])
        remove(daily_matrix_file)
        dd.dispc(' another process is doing the same stack => stopping','r','b')
    
    #init the cc_nstack matrices directly into the hdf5 file: 
    nwf  =len(md_c['t']) 
    ncpl =len(kset['md']['lat'])
    ncmp =len(in_['cc_cmp'])
    ndate=len(md_c['date1'])
    cc_nstack=ff.create_dataset('/cc_nstack',(ncmp,ncpl,ndate),dtype=in_['cc_dtype'])

    #init the daily correlation matrix dataset into the temporary h5 file :
    for ipath in range(0,ncpl) :
        for icmp in range(0,ncmp) :
            dset_name=h5_get_station_pair_tree_path(kset['md']['id'][ipath],in_['cc_cmp'][icmp])
            h5_init_dataset(ff_cc,'/cc'+dset_name,(ndate,nwf),in_['gzip'],in_['cc_dtype'])

    #now copy the cc and cc_nstack day per day into the (temporary) h5 file 
    daily_file=glob.glob(kset['out_dir']+'/*.h5') #faiblesse ici ??
    daily_file.sort()
    _indexing = 0
    for iday in daily_file :
        h5_day=h5py.File(iday,'r')
        ndate_per_day=np.shape(h5_day['cc_nstack'])[2]
        I1 = _indexing
        I2 = I1+ndate_per_day
        _indexing = I2
        cc_nstack[:,:,I1:I2]=h5_day['cc_nstack']
        #sauvegarde des correlations journaliere dans le fichier ff_cc : 
        for ipath in range(0,ncpl) :
            for icmp in range(0,ncmp) :
                dset=h5_get_station_pair_tree_path(kset['md']['id'][ipath],in_['cc_cmp'][icmp])
                if '/cc'+dset in h5_day :
                    ff_cc['/cc'+dset][I1:I2,:]=h5_day['/cc'+dset]
        h5_day.close()

    # determine the reference by summing daily correlations : 
    ref_nstack = ff.create_dataset('/ref_nstack',(ncmp,ncpl),dtype=in_['cc_dtype'])

    for ipath in range(0,ncpl) :
        perc = ipath * 100. / (ncpl-1)
        if perc % 10 == 0:
            dd.dispc(str(ipath+1) + ' / ' +  str(ncpl) + '  paths','gray','r')
        for icmp in range(0,ncmp) :
            ref_nstack[icmp,ipath] = np.sum(cc_nstack[icmp,ipath,:],axis=0)
            dset_name=h5_get_station_pair_tree_path(kset['md']['id'][ipath],in_['cc_cmp'][icmp])
            if in_['pws']:
                stack_data = pws(ff_cc['/cc'+dset_name],in_['pws_timegate'],in_['pws_power'],1/float(md_c['tau']),in_['cc_dtype'])
            elif in_['svd_wiener2']:
                stack_data = svd_wiener2(ff_cc['/cc'+dset_name],in_['svd_wiener2_m'],in_['svd_wiener2_n'],in_['svd_wiener2_nvs'],in_['cc_dtype'])
            else:
                stack_data = np.sum(ff_cc['/cc'+dset_name],axis=0)     
            if ref_nstack[icmp,ipath] :
                stack_data = stack_data/ref_nstack[icmp,ipath]
            h5_create_dataset(ff,'/ref'+dset_name,stack_data,in_['gzip'],in_['cc_dtype'])
        
    #copy daily correlations from the temporary to the final h5 file if necessary: 
    if in_['keep_event_corr'] :
        ff.copy(ff_cc['/cc'],'cc')

    #closing hdf5 file and remove the temporary file h5 and .lock file: 
    ff_cc.close()
    ff.close()
    remove(daily_matrix_file)
    rename(kset['out_file_pid'],kset['out_file'])

    # enventualy remove daily and lock files 
    if in_['remove_event_file'] : 
        for iday in daily_file : 
            remove(iday)
        removedirs(kset['out_dir'])
    remove(kset['out_file_lock_pid'])
    #displaying the stacking time : 
    dd.dispc('    stack done in '+str(int(time.time()-t0))+'s','y','n')


#-----------------------------------------------------------------------------
#                       correlate_this_set subfunctions 
#-----------------------------------------------------------------------------

def cts_open_h5_file_ev(idate,kset) :
    ''' attempt to open an hdf5 file containing data
    '''
    h5a=0 
    h5b=0 
    success=True 
    h5_set1=kset['path1']+'/'+idate['ev'].replace(':','-')+'.h5'
    h5_set2=kset['path2']+'/'+idate['ev'].replace(':','-')+'.h5'
    try : 
        h5a=h5py.File(h5_set1,'r')
    except :
        success=False 
    try :
        h5b=h5py.File(h5_set2,'r')
    except : 
        success=False 
    return h5a,h5b, success

def cts_init_h5_ev_file(kdate,in_,md,md_c,idate) :
    h5=h5py.File(kdate['out_pid'],'a')
    h5 = h5_copy_dict_to_group_ev({'in_' : in_ ,'md' : md, 'md_c' : md_c},h5)
    h5.create_dataset('idate',data=np.array([idate]))
    nwf  =len(md_c['t']) 
    ncpl =len(md['lat']) 
    ncmp =len(in_['cc_cmp'])
    if  in_['event_stack'] :    # we keep only one correlation per day = the stack of all hourly ccs:
        #h5.create_dataset('/cc',(ncmp,ncpl,1,nwf),dtype=in_['cc_dtype'])
        h5.create_dataset('/cc_nstack',(ncmp,ncpl,1),dtype=in_['cc_dtype'])
    if not in_['event_stack'] : # we keep all hourly correlations : 
        #h5.create_dataset('/cc',(ncmp,ncpl,len(in_['hr1']),nwf),dtype=in_['cc_dtype'])
        h5.create_dataset('/cc_nstack',(ncmp,ncpl,len(md_c['I1'])),dtype=in_['cc_dtype'])
    return h5 

    
def cts_get_filename_list_ev(date_list,out_dir,ev) :
    ex=[] 
    for iev in ev :
        dev={}
        dev['date']=UTCDateTime(iev.split('_')[0]).timestamp 
        dev['ev']=iev
        dev['out']=out_dir+'/'+iev.replace(':','-') +'.h5' 
        dev['out_pid']=out_dir+'/'+iev.replace(':','-') +'.'+str(os.getpid())+'.h5' 
        dev['out_lock']=dev['out'][0:-3]+'.lock'
        ex.append(dev)
    return ex

def _cts_get_current_date_as_str() : 
    ndate=UTCDateTime() 
    ndate_str=str(ndate.year)+'/'+str(ndate.month).zfill(2)+'/'+str(ndate.day).zfill(2)
    ndate_str=ndate_str+' '+str(ndate.hour).zfill(2)+':'+str(ndate.minute).zfill(2)
    return ndate_str #

def _cts_get_current_hr_as_str() : 
    ndate=UTCDateTime() 
    ndate_str=str(ndate.hour).zfill(2)+':'+str(ndate.minute).zfill(2)+':'+str(ndate.second).zfill(2)
    return ndate_str 


###########################################################################################
###########################################################################################
########    ###    ####         ####   ####   #######   ####         #####    #####    ####
########    ###    ####         ####   ####   #######   ####         ######    ####    ####
########    ###    #######   #######   ####   #######   #######   ##########   ###    #####
########    ###    #######   #######   ####   #######   #######   ###########        ######
########    ###    #######   #######   ####   #######   #######   ############      #######
########    ###    #######   #######   ####   #######   #######   ############    #########
########           #######   #######   ####        ##   #######   ###########    ##########
########           #######   #######   ####        ##   #######   ##########    ###########
###########################################################################################
###########################################################################################


def h5_copy_dict_to_group_ev(dict_,h5,prefix=''):
    for ikey in dict_ :
        h5.create_group(prefix+'/'+ikey)
        for dname, dset in dict_[ikey].items():
            dname = dname.replace('/','__')
            #if type(dset)==list and len(dset) > 0 and type(dset[0]) == dict :
            if isinstance(dset,dict):
                    h5_copy_dict_to_group_ev({dname : dset},h5,prefix=prefix+'/'+ikey)
            else :
                if type(dset)==list and len(dset) > 0 and isinstance(dset[0],str):
                    dset_tmp = []
                    for mot in dset:
                        dset_tmp.append(mot.encode())
                    h5.create_dataset(prefix+'/'+ikey+'/'+dname,data=dset_tmp)
                else:                   
                    h5.create_dataset(prefix+'/'+ikey+'/'+dname,data=dset)
    return h5 

