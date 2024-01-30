################################################
# m30_xcorr.py 
# Laurent Stehly (UGA)
# Pierre Boue (UGA)
# email: pycorr@univ-grenoble-alpes.fr
# July 2017
################################################



import pickle         
import numpy as np 
import scipy.fftpack
import scipy.io as io
import scipy.signal as signal
import scipy.linalg as linalg
import random
import h5py
import glob
import os
import inspect,sys
import time 
from obspy.core import UTCDateTime

import m_pycorr.mods.lang as lang
import m_pycorr.mods.dd as dd
import m_pycorr.m20_noise_processing as noise_processing

try : import ipdb 
except : pass 

"""
# TODO : 
# - consommation espace disque  : 
# -- on pourrait faire une 2nde version de main_loop qui distribute tous les coeurs sur le meme set :
#    avec in_['remove_daily_file'] et in_['keep_daily_corr']=False et in_['file_size']=0.1 
#    ca permettrait de limiter la consommation d'espace disque en ayant peu de fichier journaliers sur 
#    le disque en meme temps. 
#  
# -Gestion des jours sans donnees : 
# --on sauvegarde juste un fichier de correlations journaliere vide
# -- meilleurs gestion ncorr pour la normaisation des journees ...
#
# NOTES : 
# -o.s.q les traces qu'on correle ont toutes le meme vecteur temps. !
# -o.s.q tous les segments horaires correlees sont de meme taille lorsqu'on remplie le champ h5['stack']
# -le nom de chaque fichier journalier est implicite (C1_corr_normalized/xcorr_set1_set1_000/2016_01_02.mat)
# -amplitude : avant le stack : on ne normalise rien, on conserve simplement la longueur de chaque segment correle
# -la normalisation est effectuee au moment du stack. 

# CONCEPTION : 
# 
#-----------------------
# xcorr_all : define the ex dir which contains all the info we need to correlate the correlations :
#
# ex['in']     : all user input. nothing less, nothing more
# ex['out_dir']: name of the output dir ['C1_xcorre*'] 
#
# ex['md_c'] : all metadata common to all path : 
#            : date1 : date of 1st point  of *saved* cc . <= in_['date'], in_['hr'], in['hr_stack']
#            : date2 : date of last point of *saved* cc. 
#            : I1 : indice of the 1st point of each time window correlated <= in_['hr'], db['t1']
#            : I1 : indice of the 1st point of each time window correlated <= in_['hr'], db['t1']
#            : tau : sampling rate. Determined from the data 
#            : t   : time vector of the correlations <=in_['cc_maxlag'], tau 
#            : version : version number of the hdf5 file format 
#
# ex['set'] : dictionnary of the directory name of set of correlations [C1_corr_normalized/xcorr_set2_set2_000]
#           : path1 : path to the data of the 1st set of noise data to be correlated 
#           : path2 : path to the data of the 2nd set of noise data to be correlated 
#           : out_file : name of the h5 file containing all the daily correlations and the erf
#
# ex['set']['md'] : metadata of this set of correlations :
#                 : lon,lat,id, elev(ation) : numpy array of size [2,ncpl] 
# 
#---------------------------------------------------------------------------
# main_loop : - loop on each set of correlation to be computed,
#             - distribute each process across all sets of correlations
#             - decide to stack or not daily correlations ... 
#             - depends on ex['md_c'], ex['in_'], ex['set']       
#             - md_c['t'],['date1'] => used to init the h5_file                 
#
#--------------------------------------------------------------------------
# correlate_this_set : compute a single set of correlation. 1 core per day. 
#                    : For each day : open hdf5 files containing noise data 
#                    : read only traces that will be correlated (i.e that belong to this set of cc)
#                    : pre-process them 
#                    : loop on station-pair/channels couple to compute the corerlation 
#   md_c['t']   => used to init the daily h5 file  to get nwf 
#   md_c[I1,I2] => go to correlate this path 
#   in_['date'] => used to construct the daily h5 name and to loop on days
#           
#-----------------------------------------------------------------------------------            
# correlate_this_path : low lvl function that correlate a single path for a given day
#   tr0,tr1,in_[cc_maxlag'],md_c['tau'],in_[cc_dtype'],in_['cc_func]
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

def xcorr_from_firecore_station_list_multi_tag(inu,station_files,fe,cut_len) :
    ''' quickly written. To be done properly later'''
    # 
    # station_files list of station files, should be the same order as inu['path'] !!
    # 
    db={}
    for ii in range(len(inu['path'])):
        ###################
        ff=open(station_files[ii],"r")
        lines=ff.readlines()
        ###################
        itag    =inu['path'][ii]
        db[itag]=dict()
        db[itag]['in_']={'tag' : itag.split('/')[-1],'pp' :{'cut_len' :cut_len}}
        db[itag]['sta']=dict()
        for iline in lines:
            cline=iline.split(' ')
            if cline[3]=='--':
                kname=cline[1]+'_'+cline[2]+'_00'
            else:
                kname=cline[1]+'_'+cline[2]+'_'+cline[3]
            db[itag]['sta'][kname]={}
            db[itag]['sta'][kname]['name'] = cline[2]
            db[itag]['sta'][kname]['lat']  = float(cline[4])
            db[itag]['sta'][kname]['lon']  = float(cline[5])
            db[itag]['sta'][kname]['elev'] = float(cline[6])
            db[itag]['sta'][kname]['depth']= float(cline[7])
            db[itag]['sta'][kname]['dc']   = cline[0]
            db[itag]['sta'][kname]['net']  = cline[1]
            db[itag]['sta'][kname]['loc']  = cline[3] #string !
            db[itag]['sta'][kname]['kname']= kname 
        ff.close()
    ex=xcorr(inu,db,fe)    
    main_loop(ex) 


def xcorr_from_firecore_station_list(inu,station_file,fe,cut_len) :
    ''' quickly written. To be done properly later'''
    db={}
    ff=open(station_file,"r")
    lines=ff.readlines()
    db[inu['path'][0]]=dict()
    db[inu['path'][0]]['in_']={'tag' : 'set_0','pp' :{'cut_len' :cut_len}}
    db[inu['path'][0]]['sta']=dict()
    for iline in lines:
        cline=iline.split(' ')
        if cline[3]=='--':
            kname=cline[1]+'_'+cline[2]+'_00'
        else:
            kname=cline[1]+'_'+cline[2]+'_'+cline[3]
        db[inu['path'][0]]['sta'][kname]={}
        db[inu['path'][0]]['sta'][kname]['name'] = cline[2]
        db[inu['path'][0]]['sta'][kname]['lat']  = float(cline[4])
        db[inu['path'][0]]['sta'][kname]['lon']  = float(cline[5])
        db[inu['path'][0]]['sta'][kname]['elev'] = float(cline[6])
        db[inu['path'][0]]['sta'][kname]['depth']= float(cline[7])
        db[inu['path'][0]]['sta'][kname]['dc']   = cline[0]
        db[inu['path'][0]]['sta'][kname]['net']  = cline[1]
        db[inu['path'][0]]['sta'][kname]['loc']  = cline[3] #string !
        db[inu['path'][0]]['sta'][kname]['kname']= kname 
    ff.close()
    ex=xcorr(inu,db,fe)    
    main_loop(ex) 


def xcorr_from_station_list_multi_path(inu,station_file,fe,cut_len) :
    ''' quickly written. To be done properly later'''
    #station_file is a list in the sme order as path
    db={}
    for iii,iset in enumerate(inu['path']):
        ff=open(station_file[iii],"r")
        lines=ff.readlines()
        db[iset]=dict()
        set_=iset.split('/')
        if set_[-1]=='':
            set_name=set_[-2]
        else:
            set_name=set_[-1]
        db[iset]['in_']={'tag' : set_name,'pp' :{'cut_len' :cut_len}}
        db[iset]['sta']=dict()
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
        ff.close()
    ex=xcorr(inu,db,fe)
    main_loop(ex)


def xcorr_from_station_list(inu,station_file,fe,cut_len) :
    ''' quickly written. To be done properly later'''
    db={}
    ff=open(station_file,"r")
    lines=ff.readlines()
    for iset in inu['path']:
        db[iset]=dict()
        db[iset]['in_']={'tag' : 'set_0','pp' :{'cut_len' :cut_len}}
        db[iset]['sta']=dict()
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
    ff.close()
    ex=xcorr(inu,db,fe)
    main_loop(ex)

def xcorr_all_several_years(inu) : 
    ''' experimental :-) List all stations correctly if a given network (set) change for year to year. 
    Previous version would only take the station list of the 1st year in that case :/ '''
    db={}
    dd.dispc('xcorr_all_several_years: buiding year list from date list','y','b') 
    # first build a list of years from the daily date : 
    year_list = []
    for idate in inu['date'] :
        cyear = UTCDateTime(idate).year 
        if not cyear in year_list : 
                year_list.append(cyear)
    
    # main loop on all set i.e data_1hz/daily/FR,MT, ... 
    dd.dispc('  looping on set list to build the list of station from db files','c','n')
    for iset in inu['path'] :
        k = 0 
        for iyear in year_list : 
            db_file = iset+str(iyear)+'/db.pkl' 
            # check if a db file exist for this year :
            if os.path.isfile(iset+'/'+str(iyear)+'/db.pkl') :
                ff = open(db_file,'rb')
                k = k +1 
            else :
                continue 
            #if it is the first db file we just load it 
            if k == 1  :
                db_set = pickle.load(ff)
            # otherwise we add only new stations : 
            else : 
                cdb = pickle.load(ff)
                sta_already_here = db_set['sta'].keys()
                for ista in cdb['sta'].keys() :  
                    if not ista in sta_already_here : 
                        db_set['sta'][ista] = cdb['sta'][ista]
        db[iset] = db_set 
    dd.dispc('  calling xcorr to define how the correlations will be computed','c','n')
    ex = xcorr(inu,db,db[[*db][0]]['in_']['pp']['freq'])
    dd.dispc('  entering main loop','c','n')
    main_loop(ex)

    
def xcorr_all(inu) :
    # load each db file and count the number of station : 
    db={}
    for iset in inu['path']  : 
        ff = open(iset+'/'+str(UTCDateTime(inu['date'][0]).year)+'/db.pkl','rb')
        db[iset] = pickle.load(ff)
        ff.close()
    ex = xcorr(inu,db,db[[*db][0]]['in_']['pp']['freq'])
    main_loop(ex)
    

def xcorr(inu,db,fe) : 
    '''ex contain the following dict : 
    md_c     : metadata common to all cc's : cc'stime vector, all dates, tau, ... 
    set      : dict of all cc set : noise data path, outdir, station pair metadata
    in_      : input parameters of the correlation code
    out_dir  : output directory where the correlation are stored  : C1_xxx_
    '''
    date1 = int(UTCDateTime(2016,1,2,0,0,0).timestamp)
    date2 = int(UTCDateTime(2016,1,10+1,0,0).timestamp)
    in_   = {}
    in_['path']              = ['../data_5hz/daily/set1','../data_5hz/daily/set2']
    in_['path_out']          = './'
    in_['date']              = range(date1,date2,86400) 
    in_['hr1']               = range(1,22)        # if so  hour of the beginning of each segments 
    in_['hr2']               = range(2,23)        #        hour of the end of each segment
    in_['cc_maxlag']         = 600           # [s] 
    in_['cc_cmp']            = ['ZZ']        # list des channels a correler. 
    in_['cc_func']           = 'ctp_xcorr_norm'
    in_['cc_dtype']          = 'float16'     # 
    in_['cc_scaling']        = 1e12          #multiply CC by a constant (to store them in 1e12, except if xcorr_norm)
    in_['cc_tags']           = 3             # 1 : xcorr only intra-tag data,2 : xcorr only inter-tag data or 3: xcorr all  
    in_['file_size']         = 0.05          # maximum size of each h5 file ! 
    in_['gzip']              = False         # compress correlations ?
    in_['keep_daily_corr']   = True    # remove or not daily corr (i.e keep only the stack)
    in_['hr_stack']          = False         # if more than one : stack each segments or not ?
    in_['pws']               = False  # default = linear stack
    in_['pws_timegate']      = 25.
    in_['pws_power']         = 2.
    in_['svd_wiener2']       = False # apply svd-wiener2 filter before stacking Moreau et al. 2017 (GJI)
    in_['svd_wiener2_m']     = 20 # date axis, number of point for wiener window 
    in_['svd_wiener2_n']     = 20 # time lag axis, number of point for wiener window
    in_['svd_wiener2_nvs']   = None # number of singular value to keep, None will keep only singular values > 10% min value    
    in_['remove_daily_file'] = True  # rm or not daily files (keep them for debugging or if you plan to add dates
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
    ex['in_']=in_
    
    npath_per_file=ex_determine_npath_per_h5_file(in_,fe) 
    
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
                    #loc0=db[kpath1]['sta'][ksta1]['loc'] #string 
                    #loc1=db[kpath2]['sta'][ksta2]['loc'] #string 
                    #id0=db[kpath1]['sta'][ksta1]['net']+'.'+db[kpath1]['sta'][ksta1]['name']+'.'+loc0 
                    id0 = db[kpath1]['sta'][ksta1]['kname'].replace('_','.')
                    #id1=db[kpath2]['sta'][ksta2]['net']+'.'+db[kpath2]['sta'][ksta2]['name']+'.'+loc1
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
    ex['md_c']=ex_get_md_c(db,in_,fe) 
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
def main_loop(ex) : 
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
        correlate_this_set(ex['in_'],ex['md_c'],kset)
        # if not all daily correlations have been made : continue
        if ml_have_all_days_been_done(ex['in_'],kset) == False : continue 
        #if someone_stacking : continue 
        if len(glob.glob(kset['out_file'][0:-2]+'*h5'))>0 or len(glob.glob(kset['out_file_lock']))>0 : continue 
        ml_stack_and_write_this_set_in_hdf5(ex['in_'],ex['md_c'],kset) 


    # reloop in a random order : now allow several process to work in the same set :
    set_=ex['set'].values()
    set_=list(set_)
    random.shuffle(set_) 
    for kset in set_ :
        if os.path.isfile(kset['out_file']) or os.path.isfile(kset['out_file_lock']) : continue 
        dd.dispc('  2nd loop : working on '+kset['out_dir'],'c','b')
        mkdir(kset['out_dir']) # not useful 
        correlate_this_set(ex['in_'],ex['md_c'],kset) 
        # if not all daily correlations have been made : continue
        if ml_have_all_days_been_done(ex['in_'],kset) == False : continue 
        # if someone_stacking : continue 
        if len(glob.glob(kset['out_file'][0:-2]+'*h5')) or len(glob.glob(kset['out_file_lock']))>0 : continue 
        ml_stack_and_write_this_set_in_hdf5(ex['in_'],ex['md_c'],kset) 


#----------------------------------------------------------------------------------------------
def correlate_this_set(in_,md_c,kset) : 
    ''' compute a set of correlations i.e correlations stored in a XCORR/xcorr_set1_set1_000/ dir'''
    
    mkdir(kset['out_dir'])
    #some shortcuts : 
    maxlag=int(in_['cc_maxlag']/md_c['tau'])
    dtype=in_['cc_dtype']
    cc_func=in_['cc_func']
    I1=np.array(md_c['I1'],dtype='int')
    I2=np.array(md_c['I2'],dtype='int')

    # determine .mat file name containing each daily correlation :
    date_list=cts_get_filename_list(in_['date'],kset['out_dir']) 
    random.shuffle(date_list)

    #loop on each date : 
    for kdate in date_list :                     # on boucle aleatoirement sur les jours
        idate= in_['date'].index(kdate['date'])  # => il faut retrouver l'indice du jour  
        #mondain stuff 1st 
        if cts_is_this_date_already_done(kdate) :
            cts_print_already_done_message(kdate)
            continue 
        cts_print_welcome_message(kdate)
        create_lock_file(kdate['out_lock'])
        #initialize the h5 file containing the cc for this days in any case :
        h5_daily=cts_init_h5_daily_file(kdate,in_,kset['md'],md_c,idate) 
        #open h5 files containing noise data, w/o loading it into memory 
        h5a, h5b, success =  cts_open_h5_file(kdate,kset) 
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
            for icpl, kcpl in enumerate(kset['md']['id']) : 
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
                    if in_['hr_stack']     :
                        #on suppose ici que chaque segement correle a la meme longueur :
                        h5_daily['cc_nstack'][icmp,icpl,0]=sum(ncorr) #(I2[0]-I1[0])*md_c['tau']*sum(ncorr)/86400.0
                        dset_name=h5_get_station_pair_tree_path(kcpl,kcmp)
                        if in_['pws']:
                            h5_create_dataset(h5_daily,'/cc'+dset_name,pws(cc_hr,in_['pws_timegate'],in_['pws_power'],1/float(md_c['tau']),in_['cc_dtype']),in_['gzip'],in_['cc_dtype'])
                        elif in_['svd_wiener2']:
                            h5_create_dataset(h5_daily,'/cc'+dset_name,svd_wiener2(cc_hr,in_['svd_wiener2_m'],in_['svd_wiener2_n'],in_['svd_wiener2_nvs'],in_['cc_dtype']),in_['gzip'],in_['cc_dtype'])           
                        else :
                            h5_create_dataset(h5_daily,'/cc'+dset_name,cc_hr.sum(axis=0),in_['gzip'],in_['cc_dtype'])
                    if not(in_['hr_stack']):  
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


#------------------------------------------------------------------------
def correlate_this_path(trace1,trace2,maxlag,I1,I2,cc_type,data_type) : 
    ''' correlate a single path at a give date. Low lvl functions only'''
    fh = globals()[cc_type]
    trace1[np.isnan(trace1)] = 0.0
    trace2[np.isnan(trace2)] = 0.0
    trace01s = [trace1[I1[itr]:I2[itr]]  for itr in range(0,len(I1),1)]
    trace02s = [trace2[I1[itr]:I2[itr]]  for itr in range(0,len(I1),1)]
    maxlag   = int(maxlag)
    N        = len(trace01s)
    corr     = np.zeros([N,2*maxlag+1], dtype=data_type)
    ncorr    = np.zeros(N)
    for ihr in range(N):
        if trace01s[ihr].std() > 1e-15:      #  numpy.max(numpy.abs(trace01s[i])) >= 1e-20:
            if trace02s[ihr].std() > 1e-15:  #  numpy.max(numpy.abs(trace02s[i])) >= 1e-20:
                #try:
                corr[ihr] = fh(trace02s[ihr], trace01s[ihr],maxlag).astype(data_type)
                ncorr[ihr] = 1 # TO BE MODIFIED !!! ncorr[ihr] = 1 or 0
                #except:pass
    return corr,ncorr



#------------------------------------------------------------------------------------
#                                 scheduler subfunctions 
#------------------------------------------------------------------------------------
def ex_get_md_c(db,in_,fe) : 
    md_c={}
    md_c['tau']  =1./fe #db[[*db][0]]['in_']['pp']['freq']    #
    
    #info sur les donnes de bruit
    time={}
    time['fe']   = 1./md_c['tau'] 
    time['t1']   = db[[*db][0]]['in_']['pp']['cut_len'] # t1 en [s] 
    time['t2']   = 86400-time['t1']-1             
    time['npts'] =time['fe']*(86400-time['t1']-time['t1'])          #
    time['time']= np.arange(time['t1'],time['t2'],1.0/time['fe'])   # in second w.r to midnight
    ds1=np.array(in_['hr1'])*3600-time['t1'] # difference de temps [s]
    ds2=np.array(in_['hr2'])*3600-time['t1'] # entre chaque fen et le t1 des traces

    #keep the indice of each hourly slice :
    md_c['I1']=np.round(ds1*time['fe']).astype('int')                 # idem ms en nb de points
    md_c['I2']=np.round(ds2*time['fe']).astype('int')
    if md_c['I2'][-1] > time['npts'] : 
        dd.dispc('  ex_get_h5_time_vector : will try to correlate time that do not exist','r','b')

    md_c['version']= 0.9 
    #md_c['t']= np.arange(-in_['cc_maxlag'],in_['cc_maxlag']+md_c['tau'],md_c['tau'])
    md_c['t'] = np.linspace(-in_['cc_maxlag'],in_['cc_maxlag'],int(2*in_['cc_maxlag']*1/md_c['tau']+1))

    #add the list of components from in_ : so that it cab be changed later if we rotate the CC : 
    md_c['cmp']=in_['cc_cmp']

    #construct date vector : 
    md_c['date1']=[]
    md_c['date2']=[]
    if in_['hr_stack'] :
        for idate in in_['date'] :
            md_c['date1'].append((UTCDateTime(idate)+time['t1']).timestamp)
            md_c['date2'].append((UTCDateTime(idate)+time['t2']).timestamp)
    if not in_['hr_stack'] :
        for idate in in_['date'] :
            for ihr in range(0,len(in_['hr1'])) :
                md_c['date1'].append((UTCDateTime(idate)+(in_['hr1'][ihr]*3600)).timestamp)
                md_c['date2'].append((UTCDateTime(idate)+(in_['hr2'][ihr]*3600)).timestamp)
    return md_c

def ex_determine_npath_per_h5_file(in_,fe) :
    size_single_path = (in_['cc_maxlag']*2.0+1.0) * fe * len(in_['cc_cmp'])
    if in_['cc_dtype']=='float64'  : size_single_path = size_single_path * 8.0
    elif in_['cc_dtype']=='float32'  : size_single_path = size_single_path * 4.0
    elif in_['cc_dtype']=='float16'  : size_single_path = size_single_path * 2.0
    else : 
        dd.dispc(' cc_dtype different from float16/32/64','y','r')
        
    ndate=1
    if in_['keep_daily_corr'] :
        ndate = ndate *len(in_['date'])
        if not in_['hr_stack'] :
            ndate = ndate  *len(in_['hr1'])
    size_single_path=size_single_path*ndate/1024./1024./1024. #in Go
    npath_per_file=int(in_['file_size']/size_single_path)
    dd.dispc('  we have '+str(ndate)+' dates to save and '+str(len(in_['cc_cmp']))+' cmp','y','b')
    dd.dispc('  => we will put at most '+str(npath_per_file)+' paths per h5 file','y','b')
    return npath_per_file 



#------------------------------------------------------------------------------
#                            main loop subfunctions 
#_-----------------------------------------------------------------------------
def ml_have_all_days_been_done(in_,kset) : 
    date_list=cts_get_filename_list(in_['date'],kset['out_dir']) 
    for idate in date_list : 
        if os.path.isfile(idate['out'])== False :
            return False 
    return True             


def ml_stack_and_write_this_set_in_hdf5(in_,md_c,kset) :
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
    h5_copy_dict_to_group({'in_':in_, 'md':kset['md'], 'md_c' : md_c},ff)

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
    for iday in daily_file :
        h5_day=h5py.File(iday,'r')
        ndate_per_day=np.shape(h5_day['cc_nstack'])[2]
        I1=h5_day['idate'][0]*ndate_per_day
        I2=I1+ndate_per_day
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
            #if stack_data.any():
            h5_create_dataset(ff,'/ref'+dset_name,stack_data,in_['gzip'],in_['cc_dtype'])
        
    #copy daily correlations from the temporary to the final h5 file if necessary: 
    if in_['keep_daily_corr'] :
        ff.copy(ff_cc['/cc'],'cc')

    #closing hdf5 file and remove the temporary file h5 and .lock file: 
    ff_cc.close()
    ff.close()
    remove(daily_matrix_file)
    rename(kset['out_file_pid'],kset['out_file'])

    # enventualy remove daily and lock files 
    if in_['remove_daily_file'] : 
        for iday in daily_file : 
            remove(iday)
        removedirs(kset['out_dir'])
    remove(kset['out_file_lock_pid'])
    #displaying the stacking time : 
    dd.dispc('    stack done in '+str(int(time.time()-t0))+'s','y','n')


def pws(data,timegate,power,fe,frmt) :
    stack    = np.zeros(data.shape[1], dtype=frmt)
    c = np.zeros(data.shape[1], dtype='c16')
    for trace in data:
        if trace.any():
            trace -= trace.mean()
            c += np.exp(1j*np.angle(signal.hilbert(trace.astype('float32'))))
    c   = np.abs(c)/data.shape[0]
    box = int(timegate * fe)
    if not box: dd.dispc('pws timegate too small !','r','b')
    s_c = np.convolve(signal.boxcar(box) / box, c, 'same')
    s_c = np.power(s_c, power)
    for trace in data:
        stack += c * trace
    return stack.astype(frmt)

#def pws_up(data,dic_args,fe,frmt) :
#    timegate = dic_args['timegate']
#    power    = dic_args['power']
#    stack    = np.zeros(data.shape[1], dtype=frmt) 
#    phasewin = pi/(data.shape[1]+1);
#    c = np.zeros(data.shape[1], dtype='c16')
#    iphase = np.zeros(data.shape, dtype=frmt)
#    for ind,trace in enumerate(data):
#        trace -= trace.mean()
#        iphase[ind,:] = np.angle(signal.hilbert(trace))
#    ...
#    c   = np.abs(c)/data.shape[0]
#    box = int(timegate * fe)
#    s_c = np.convolve(signal.boxcar(box) / box, c, 'same')
#    s_c = np.power(s_c, power)
#    for trace in data:
#        stack += c * trace
#    return stack

def svd_wiener2(data,m,n,nvs=None,frmt='float32') :
    if len(data.shape)==1:
        return signal.wiener(data, mysize=n).astype(frmt)
    else:
        [U,S,V] = linalg.svd(data)
        if nvs is not None:
            if nvs > min(data.shape):
                nvs=min(data.shape)
        else:
            #Sdiff2 = np.abs(np.diff(np.diff(20*np.log10(S/max(S)))))
            #nvs = np.where(Sdiff2==max(Sdiff2))[0][0]
            nvs = len(np.where(20*np.log10(S/max(S))>0.1*min(20*np.log10(S/max(S))))[0])
        Xwiener = np.zeros(data.shape)
        for kk in range(int(nvs)):
            X = S[kk]*np.outer(U[:,kk],V[kk,:])
            Xwiener = signal.wiener(X, mysize=[m,n]) + Xwiener
        return signal.wiener(Xwiener, mysize=[5,5]).mean(axis=0).astype(frmt)

def mplot(data) :
    import matplotlib.pyplot as plt
    if len(data.shape)==2:
        plt.matshow(data);
        plt.colorbar()
    else: 
        plt.plot(data)
    plt.gca().set_aspect('auto', adjustable='box')
    plt.show()


#-----------------------------------------------------------------------------
#                       correlate_this_set subfunctions 
#-----------------------------------------------------------------------------
def cts_preprocess_data(h5_data,pp_list,pp_args,tau) :
    ''' loop on each trace of dict of numpy array (h5_data) and preprocess them 
        calling m10_noise_processing functions'''
    for iid,itr in h5_data.items() : 
        for ipp, ipp_name  in enumerate(pp_list):        
            h5_data[iid],_out=getattr(noise_processing,ipp_name)(itr,1.0/tau,pp_args[ipp])
    return h5_data


def cts_read_data(h5_file,id_,ch,nset) : 
    ''' read only noise data that will be correlated from an open h5_file  '''
    ch_list=[]
    for ich in ch : 
        ch_list.append(ich[nset])
    ch_list=set(ch_list)
    noise_data={}
    for iid in id_[:,nset] :
        aa=iid.decode('utf8').split('.')
        for icmp in ch_list :
            #h5_node  = iid+'/'+icmp
            h5_node  ='/'+aa[0]+'/'+aa[1]+'.'+aa[2]+'/'+icmp
            dest_key=iid.decode('utf8')+'.'+icmp
            if h5_node in  h5_file  :
                noise_data[dest_key]=h5_file[h5_node][:].flatten()
    return noise_data


def cts_open_h5_file(idate,kset) :
    ''' attempt to open an hdf5 file containing noise data. We reconstruct the path to the file  
    using kset['path'] + year + julian day 
    '''
    h5a=0 
    h5b=0 
    success=True 
    cdate=UTCDateTime(idate['date'])
    h5_set1=kset['path1']+'/'+str(cdate.year)+'/day_'+str(cdate.julday).zfill(3)+'.h5'
    h5_set2=kset['path2']+'/'+str(cdate.year)+'/day_'+str(cdate.julday).zfill(3)+'.h5'
    try : 
        h5a=h5py.File(h5_set1,'r')
    except :
        success=False 
    try :
        h5b=h5py.File(h5_set2,'r')
    except : 
        success=False 
    return h5a,h5b, success

def cts_init_h5_daily_file(kdate,in_,md,md_c,idate) :
    h5=h5py.File(kdate['out_pid'],'a')
    h5 = h5_copy_dict_to_group({'in_' : in_ ,'md' : md, 'md_c' : md_c},h5)
    h5.create_dataset('idate',data=np.array([idate]))
    nwf  =len(md_c['t']) 
    ncpl =len(md['lat']) 
    ncmp =len(in_['cc_cmp'])
    if  in_['hr_stack'] :    # we keep only one correlation per day = the stack of all hourly ccs:
        #h5.create_dataset('/cc',(ncmp,ncpl,1,nwf),dtype=in_['cc_dtype'])
        h5.create_dataset('/cc_nstack',(ncmp,ncpl,1),dtype=in_['cc_dtype'])
    if not in_['hr_stack'] : # we keep all hourly correlations : 
        #h5.create_dataset('/cc',(ncmp,ncpl,len(in_['hr1']),nwf),dtype=in_['cc_dtype'])
        h5.create_dataset('/cc_nstack',(ncmp,ncpl,len(in_['hr1'])),dtype=in_['cc_dtype'])
    return h5 


def cts_is_this_date_already_done(ifl) :
    if os.path.isfile(ifl['out']) | os.path.isfile(ifl['out_lock']) : return True 
    if len(glob.glob(ifl['out'][0:-3]+'.*h5'))>0 : return True 
    return False 

    
def cts_get_filename_list(date_list,out_dir) : 
    ex=[] 
    for idate in date_list :
        cdate=UTCDateTime(idate) 
        cdate_str=str(cdate.year)+'_'+str(cdate.month).zfill(2)+'_'+str(cdate.day).zfill(2)
        ddate={}
        ddate['date']=idate 
        ddate['out']=out_dir+'/'+cdate_str+'.h5' 
        ddate['out_pid']=out_dir+'/'+cdate_str+'.'+str(os.getpid())+'.h5' 
        ddate['out_lock']=ddate['out'][0:-3]+'.lock'
        ex.append(ddate) 
    return ex


def cts_print_already_done_message(kdate) : 
    ndate_str = _cts_get_current_date_as_str()
    str_='      : '+ndate_str+' : '+kdate['out']+' : already (being) done'
    dd.dispc(str_,'r','n')
    

def cts_print_welcome_message(kdate) :
    ndate_str = _cts_get_current_date_as_str()
    str_='      : '+ndate_str+' : '+kdate['out']
    dd.dispc(str_,'c','n')


def _cts_get_current_date_as_str() : 
    ndate=UTCDateTime() 
    ndate_str=str(ndate.year)+'/'+str(ndate.month).zfill(2)+'/'+str(ndate.day).zfill(2)
    ndate_str=ndate_str+' '+str(ndate.hour).zfill(2)+':'+str(ndate.minute).zfill(2)
    return ndate_str 

def _cts_get_current_hr_as_str() : 
    ndate=UTCDateTime() 
    ndate_str=str(ndate.hour).zfill(2)+':'+str(ndate.minute).zfill(2)+':'+str(ndate.second).zfill(2)
    return ndate_str 



##########################################################################################
##########################################################################################
###########         ####            ####           ######           ######################
##########          ####            ####            #####            #####################
##########     #########    ####    ####    ####    #####    ####    #####################
#########     ##########    ####    ####    ####   ######    ####   ######################
#########     ##########    ####    ####           ######           ######################
#########      #########    ####    ####    ####    #####    ####    #####################
##########          ####            ####    ####    #####    ####    #####################
###########         ####            ####    #####    ####    #####    ####################
##########################################################################################
##########################################################################################

#--------------------------------------------------------------------------------------------
#                          correlate_this_path (for this day) subfunctions 
#--------------------------------------------------------------------------------------------
def ctp_xcorr_norm(trace01, trace02,maxlag): # cross-coherence Time normalized before Corr
    lentrace   = len(trace01)
    goodnumber = int(maxlag+lentrace)
    tr2 = np.zeros(goodnumber)
    tr2[0:lentrace] = trace02
    tr2[0:lentrace] /= np.sqrt(np.sum(tr2[0:lentrace]**2))
    tr2 = scipy.fftpack.fft(tr2,overwrite_x=True)
    tr2.imag *= -1
    tr1 = np.zeros(goodnumber)
    tr1[maxlag:int(maxlag+lentrace)]= trace01
    tr1[maxlag:int(maxlag+lentrace)] /= np.sqrt(np.sum(tr1[maxlag:int(maxlag+lentrace)]**2))
    tr2 *= scipy.fftpack.fft(tr1,overwrite_x=True)
    tr2[np.isnan(tr2)] = 0.0+0.0j;
    tr2[np.isinf(tr2)] = 0.0+0.0j;
    return (scipy.fftpack.ifft(tr2,overwrite_x=True)[0:2*maxlag+1].real)

def ctp_xcorr(trace01, trace02,maxlag): # cross-correlation
    lentrace=len(trace01)
    goodnumber = int(maxlag+lentrace)
    tr2 = np.zeros(goodnumber)
    tr2[0:lentrace] = trace02
    tr2 = scipy.fftpack.fft(tr2,overwrite_x=True)
    tr2.imag *= -1
    tr1 = np.zeros(goodnumber)
    tr1[maxlag:int(maxlag+lentrace)]= trace01
    tr2 *= scipy.fftpack.fft(tr1,overwrite_x=True)
    tr2[np.isnan(tr2)] = 0.0+0.0j;
    tr2[np.isinf(tr2)] = 0.0+0.0j;
    return (scipy.fftpack.ifft(tr2,overwrite_x=True)[0:2*maxlag+1].real)

def ctp_coher(trace01, trace02,maxlag): # cross-coherence
    lentrace   = len(trace01)
    goodnumber = int(maxlag+lentrace)
    tr2 = np.zeros(goodnumber)
    tr2[0:lentrace] = trace02
    tr2 = scipy.fftpack.fft(tr2,overwrite_x=True)
    tr2.imag *= -1
    tr1 = np.zeros(goodnumber)
    tr1[maxlag:int(maxlag+lentrace)]= trace01
    tr1 = scipy.fftpack.fft(tr1,overwrite_x=True)
    ## wtlvl
        #wlvl = 1.0
        #temp2 = numpy.absolute(tr2)
        #temp2[numpy.isnan(temp2)] = 0.0;
        #temp2[numpy.isinf(temp2)] = 0.0;
        #mtemp2 = numpy.amax(temp2) * wlvl / 100
        #numpy.putmask(temp2,temp2 <= mtemp2, mtemp2)
        #temp1 = numpy.absolute(tr1)
        #temp1[numpy.isnan(temp1)] = 0.0;
        #temp1[numpy.isinf(temp1)] = 0.0;
        #mtemp1 = numpy.amax(temp1) * wlvl / 100
        #numpy.putmask(temp1,temp1 <= mtemp1, mtemp1)
    ##
    tr2 = (tr1 * tr2) / (np.absolute(tr1) * np.absolute(tr2))
    #tr2 = (tr1 * tr2) / (temp1 * temp2)
    tr2[np.isnan(tr2)] = 0.0+0.0j;
    tr2[np.isinf(tr2)] = 0.0+0.0j;
    return (scipy.fftpack.ifft(tr2,overwrite_x=True)[0:2*maxlag+1].real)

def ctp_coher_tap(trace01, trace02,maxlag): # cross-coherence + taper
    taper      = 0.15
    strength   = 1000
    lentrace   = len(trace01)
    goodnumber = int(maxlag+lentrace)
    tr2 = np.zeros(goodnumber)
    tr2[0:lentrace] = trace02
    tr2 = scipy.fftpack.fft(tr2,overwrite_x=True)
    tr2.imag *= -1
    tr1 = np.zeros(goodnumber)
    tr1[maxlag:maxlag+lentrace]= trace01
    tr1 = scipy.fftpack.fft(tr1,overwrite_x=True)
    tt = np.append(signal.tukey(divmod(len(tr1),2)[0],taper),signal.tukey(divmod(len(tr1),2)[0] + divmod(len(tr1),2)[1],taper))
    tt = 1+(1-tt)*strength
    tr2 = (tr1 * tr2) / (np.absolute(tr1) * np.absolute(tr2) *tt)
    tr2[np.isnan(tr2)] = 0.0+0.0j;
    tr2[np.isinf(tr2)] = 0.0+0.0j;
    return (scipy.fftpack.ifft(tr2,overwrite_x=True)[0:2*maxlag+1].real)

def ctp_deconv(trace01, trace02, maxlag): # deconvolution
    ''' voir la signification d'EXTRA!!!'''
    EXTRA      = 1
    lentrace   = len(trace01)
    goodnumber = int(maxlag+lentrace)
    tr2 = np.zeros(goodnumber)
    tr2[0:lentrace] = trace02
    tr2 = scipy.fftpack.fft(tr2,overwrite_x=True)
    tr2.imag *= -1
    tr1 = np.zeros(goodnumber)
    tr1[maxlag:maxlag+lentrace]= trace01
    tr1 = scipy.fftpack.fft(tr1,overwrite_x=True)
    ################################################################
    ##### WATER LVL
    #
    ############
    #wlvl = EXTRA
    ############
    #temp = numpy.absolute(tr2)
    #mtemp = numpy.amax(temp)* wlvl / 100
    #numpy.putmask(temp,temp <= mtemp, mtemp)
    #tr2 = (tr1 * tr2) / (temp ** 2)
    ################################################################
    ##### NOISE PADDING
    #tr2_n = (numpy.random.random(goodnumber)-0.5)*2*(numpy.std(trace02)*EXTRA/100)
    tr2_n = (np.random.random(goodnumber)-0.5)*2*(EXTRA)
    tr2_n[0:lentrace] = trace02
    tr2_n = scipy.fftpack.fft(tr2_n,overwrite_x=True)
    tr2_n.imag *= -1
    tr2 = (tr1 * tr2) / (np.absolute(tr2_n)**2)
    #################################################################
    #################################################################
    ##### SMOOTHING
    #temp = numpy.absolute(tr2)
    ###########
    #window_len = EXTRA
    ###########
    #s = numpy.r_[2*temp[0]-temp[window_len-1::-1],temp,2*temp[-1]-temp[-1:-window_len:-1]]
    ##s = numpy.r_[numpy.zeros(len(temp[window_len-1::-1])),temp,numpy.zeros(len(temp[-1:-window_len:-1]))]
    #w = numpy.ones(window_len,'d')
    #y = numpy.convolve(w/numpy.sum(w),s,mode='same')
    #temp = y[window_len:-window_len+1]
    #tr2 = (tr1 * tr2) / (temp**2)
    #################################################################
    #################################################################
    return (scipy.fftpack.ifft(tr2,overwrite_x=True)[0:2*maxlag+1].real)


def ctp_xcorr_norm_envelope(trace01, trace02,maxlag): 
    # cross-coherence Time normalized before Corr
    # since the 'envelope' is always larger than zero we need add a demean to make the correlation meaningful
    # THIS IS TOO SIMPLE TO BE EXPLAINED!!!!
    trace01 = signal.detrend(trace01, type='constant')
    trace02 = signal.detrend(trace02, type='constant')
    lentrace   = len(trace01)
    goodnumber = int(maxlag+lentrace)
    tr2 = np.zeros(goodnumber)
    tr2[0:lentrace] = trace02
    tr2[0:lentrace] /= np.sqrt(np.sum(tr2[0:lentrace]**2))
    tr2 = scipy.fftpack.fft(tr2,overwrite_x=True)
    tr2.imag *= -1
    tr1 = np.zeros(goodnumber)
    tr1[maxlag:int(maxlag+lentrace)]= trace01
    tr1[maxlag:int(maxlag+lentrace)] /= np.sqrt(np.sum(tr1[maxlag:int(maxlag+lentrace)]**2))
    tr2 *= scipy.fftpack.fft(tr1,overwrite_x=True)
    c12 = (scipy.fftpack.ifft(tr2,overwrite_x=True)[0:2*maxlag+1].real)
    return c12


##########################################################################################
##########################################################################################
#######    ###    ####         ####   ####   #######   ####         #####    #####    ####
#######    ###    ####         ####   ####   #######   ####         ######    ####    ####
#######    ###    #######   #######   ####   #######   #######   ##########   ###    #####
#######    ###    #######   #######   ####   #######   #######   ###########        ######
#######    ###    #######   #######   ####   #######   #######   ############      #######
#######    ###    #######   #######   ####   #######   #######   ############    #########
#######           #######   #######   ####        ##   #######   ###########    ##########
#######           #######   #######   ####        ##   #######   ##########    ###########
##########################################################################################
##########################################################################################
#---------------------------------------------------------------------------------------
#                       utility functions 
#----------------------------------------------------------------------------------------        
def h5_create_dataset(h5_file,dset_name,data,gzip,frmt) : 
    if gzip == True :
        h5_file.create_dataset(dset_name,data=data,dtype=frmt,compression="gzip", compression_opts=9)
    else : 
        h5_file.create_dataset(dset_name,data=data,dtype=frmt)
    return h5_file 

def h5_init_dataset(h5_file,dset_name,data,gzip,frmt) : 
    if gzip == True :
        h5_file.create_dataset(dset_name,data,dtype=frmt,compression="gzip", compression_opts=9)
    else : 
        h5_file.create_dataset(dset_name,data,dtype=frmt)

def h5_get_station_pair_tree_path(id_,cmp) :
    dset='/'+id_[0].decode('utf8')+'/'+id_[1].decode('utf8')+'/'+cmp
    return dset

def h5_copy_dict_to_group(dict_,h5,prefix=''):
    for ikey in dict_ :
        h5.create_group(prefix+'/'+ikey)
        for dname, dset in dict_[ikey].items():
            if type(dset)==list and len(dset) > 0 and type(dset[0]) == dict :
                    h5_copy_dict_to_group({dname : dset[0]},h5,prefix='/'+ikey)
            else :
                try:
                    if type(dset)==list and len(dset) > 0 and isinstance(dset[0],str):
                        dset_tmp = []
                        for mot in dset:
                            dset_tmp.append(mot.encode())
                        h5.create_dataset(prefix+'/'+ikey+'/'+dname,data=dset_tmp)
                    else:                    
                        h5.create_dataset(prefix+'/'+ikey+'/'+dname,data=dset)
                except:
                    h5.create_dataset(prefix+'/'+ikey+'/'+dname,data='None')
    return h5 


def create_lock_file(filename) : 
    ff=open(filename,'wb')
    pickle.dump(filename,ff)
    ff.close()
    

def mkdir(out_dir) : 
    if os.path.isdir(out_dir) == False :
        try : 
            os.makedirs(out_dir)
        except  :
            pass 

def rename(src,dest) : 
    try : 
        os.rename(src,dest) 
    except  :
        pass 

def remove(file_name) : 
    try :
        os.remove(file_name) 
    except :
        pass 

def removedirs(dir_name) :
    try :
        os.removedirs(dir_name)
    except : 
        pass 
