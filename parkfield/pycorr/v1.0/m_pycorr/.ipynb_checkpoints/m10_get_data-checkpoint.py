################################################
# m10_get_data.py 
# Pierre Boue (UGA)
# Laurent Stehly (UGA)
# email: pycorr@univ-grenoble-alpes.fr
# May 2017
# last big update : June 2019
################################################
"""
=================================
get data from FDSN webservices  
=================================
##### AVAILABLE DATACENTERS 
# BGR     http://eida.bgr.de
# ETH     http://eida.ethz.ch
# GEONET  http://service.geonet.org.nz
# GFZ     http://geofon.gfz-potsdam.de
# INGV    http://webservices.rm.ingv.it
# IPGP    http://eida.ipgp.fr
# IRIS    http://service.iris.edu
# KOERI   http://eida.koeri.boun.edu.tr
# LMU     http://erde.geophysik.uni-muenchen.de
# NCEDC   http://service.ncedc.org
# NEIP    http://eida-sc3.infp.ro
# NERIES  http://www.seismicportal.eu
# ODC     http://www.orfeus-eu.org
# ORFEUS  http://www.orfeus-eu.org
# RESIF   http://ws.resif.fr
# SCEDC   http://service.scedc.caltech.edu
# USGS    http://earthquake.usgs.gov
# USP     http://sismo.iag.usp.br
# ....



# HOW METADATA ARE FILLED 
# metadata are stored in "md" dict which is saved day per day in the _metadata directory. 
# Unfortunately md is filled by two functions : download and download=>load_from_source=>check_if_trace_correct
#
# For each day/station/channel, the following fields are filled by check_if_trace_correct : 
# -- [nsgement] : number of miniseed segments, or -1 if server not reachable or timeout while processing 
# -- [duration] : duration of the trace [s]  , or -1 if server not reachable or timeout while processing 
# -- [correct_sampling_rate] : True if all mseed segements have the same sampling_rate, False otherwise
# -- [correct_nsegment]      : True if there is less than in_['qc']['max_nsegments'] segments 
# -- [correct_duration]      : True if there is more than in_['qc']['min_duration'] seconds of signal 
# -- [nan_detected]          : True (trace rejected) if any nan on trace, False otherwise
# -- [everything_correct]    : True if the trace integrity is fine before preprocessing. 
#
# if below the radar : it displays "weird error during ..."
#
# -- [output] : filled by check_if_trace_correct if we could reach the server, and we did not time_out :
#                                                => 'OK',' no data', 'data found but incorrect trace'
#              : filled by download if did not reach the server or got a time out : 
#                                                => 'could not initialize the client', 'time_out'
#
# The following fields are filled only by download : 
# -- [success]  = True if we could download/process and save the data, False otherwise. 
# -- [time_out] = True if we reach a timeout, false otherwise.
#
#=> To know if everything went ok :=> ['success'] or ['output'] for a string
#=> To know if the trace was correct before pre-processing : ['everything_correct']
#
#------------------------------------------------------------
# FUNCTIONS  : 
#..............................................................
# configure : configure which/how data will be download. Output db.pkl file in data_5.0hz/daily/2016/FR/db.mat
#  -get_network_table 
#  -read_station_list
#  -write_db_file 
#
#...............................................................
# download : main function that acutally download data.
#   loop on day/station/compnonent 
#     create lock_file
#     read_metadata '.../_metadata/day_001.pkl'
#     has_this_ch_been_already_downladed ?
#     initialize_metadata
#     if data_center = 'SDS' else initialize client 
#     run subprocess to download and process data 
#     update metadata field in the .h5 file ()('/_metadata/fe',...)
#     update metadata file (_metadata/day_001.h5) 
#     clean everything : delete lock file, ... 
#
#...............................................................................
# load_from_source : download and process a single day/station/cmp and write it in the h5 file
#   init obspy.stream object 
#   if client ='SDS'
#     determine_ch_and_locid_from_archive | read_SDS
#   else : 
#     determine_ch_and_locid_from_client | client_get_waveforms 
#   check_if_trace_correct
#   if everything_correct : 
#     process_trace() | check_if_trace_correct 
#   return 
#
#...................................................................................
# determine_ch_and_locid_from_archive
#  init obspsy.stream object 
#  loop on channel [BH/HH]
#    try : read_sds, select_station 
# 
#...................................................................................
# check_if_trace_correct()
#   is_there_some_data
#   compute_total_duration_of_trace
#   check_number_of_segment_is_ok 
#   is_there_some_nan 
#   check_sampling_rate
#   check_total_duration 
#
"""
import os 
import glob
import time
import copy
import pickle
import random
import h5py as h5 
import numpy as np
import scipy.io as io
import obspy.clients.fdsn
import multiprocessing
import m_pycorr.mods.lang as lang
import m_pycorr.mods.dd as dd
from obspy.core import UTCDateTime, Stream
from obspy.core.inventory.inventory import read_inventory
from obspy import read


try : 
    import ipdb
except :
    pass 

# NCEDC cannot be reached, fixed by
from obspy.clients.fdsn.header import URL_MAPPINGS
URL_MAPPINGS['NCEDC'] = "https://service.ncedc.org"


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



def configure(input_user={}):
    ''' this function is used configure the data download. It should be called before download() 
    It create the ouput directory where the noise records in h5 format are stored (../data_5hz/daily/EU/2011/)
    and put a db.mat and db.pkl file which contains the configuration of the downloaded. 
    This db file is then used by download to get the data !
    '''
    in_=dict()
    in_['ch']                  = ['BH','HH']            # channel that we will attempt to download by descending priority 
    in_['cmp']                 = ['Z']                  # list of component we will download ([Z,N,E]) 
    in_['tag']                 = 'EU'                   # subdir were the data will be saved 
    in_['day1']                = UTCDateTime(2000,1,1)  # date range
    in_['day2']                = UTCDateTime(2000,1,2)  # 
    in_['format']              = 'h5'                   # only .h5 is supported 
    in_['station_list']        = 'station.txt'          # text file containing the list of station to be downloaded
    in_['path_out']            = './'                   # output path
    in_['sds_path']            = '/media/resif/validated_seismic_data/'
    in_['sds_metapath']        = '/media/resif/metadata/stationXML/'
    in_['user_id']             = None                   # user id for fdsn client
    in_['password']            = None                   # password for fdsn client
    in_['pp']                  = dict()                 # preprocessing  options : 
    in_['pp']['remove_resp']   = True                   # remove instrumental response ? not even loaded if SDS
    in_['pp']['cut_len']       = 0                      # old param
    in_['pp']['tap_len']       = 15*60                  #  - taper lenght in seconds, dwld request : 2*tap_len + 86400s  
    in_['pp']['freq']          = 5                      #  - frequency. to which data are decimated [hz] 
    try :
        in_['pp']['f_prefilt']     = (0.008, input_user['pp']['freq'] / 2 - 0.05 * input_user['pp']['freq'])
    except :
         in_['pp']['f_prefilt']     = (0.008, in_['pp']['freq'] / 2 - 0.05 * in_['pp']['freq'])
    in_['pp']['glitch']        = True                   # should we apply glitch correction ?
    in_['qc']                  = dict()                 # quality control options :
    in_['qc']['max_nsegments'] = 8       # - max. number of segement in each mseed file = max number of gaps 
    in_['qc']['min_duration']  = 72000   # - minimum length of signals per days 72000s=20h [s]
    in_['qc']['timeout']       = 300     # - maximum time to retrieve and process a single day [s]
    in_['ev']                  = {} # always '{}'
    in_['ev']['only_event']    = False # default False
    in_['ev']['event_list']    = 'events.txt' # input station file from 00_get_events.py or other
    in_['ev']['time_before']   = 0 # [s] from source time
    in_['ev']['time_after']    = None # [s] from source time, if None, autoscale from magnitude
    in_                        = lang.parse_options(in_,input_user)
    print('###### ALL')
    dd.dd(dict(in_))
    print('###### PP')
    dd.dd(dict(in_['pp']))
    print('###### QC')
    dd.dd(dict(in_['qc']))
    print('###### EV')
    dd.dd(dict(in_['ev']))    
    net_table = get_network_table()
    sta       = read_station_list(in_['station_list'])
    for kname in sta :
        if sta[kname]['dc'] == '--':
            sta[kname]['dc'] = net_table[sta[kname]['net']]
    if in_['ev']['only_event']:
        ev        = read_event_list(in_['ev']['event_list'])
        out_dir   = in_['path_out'] + '/data_'+str((in_['pp']['freq']))+'hz/events/'+in_['tag']
        mkdir(out_dir)
        write_db_file(out_dir,in_,sta,ev)
    else:
        year      = str(UTCDateTime(in_['day1']).year)
        out_dir   = in_['path_out'] + '/data_'+str((in_['pp']['freq']))+'hz/daily/'+in_['tag']+'/'+year
        mkdir(out_dir)
        write_db_file(out_dir,in_,sta)





##########################################################################################
##########################################################################################
########          #####    ################    ###    ###########          ###############
########           #####    ######  ######    ####    ###########           ##############
########   ####     ####    ######  ######    ####    ###########   ####     #############
########   #####    #####    ####    #####   #####    ###########   #####    #############
########   #####    #####    ####    ####    #####    ###########   #####    #############
########   ####     ######    ##  #   ##    ######    ###########   ####     #############
########           #######        ##        ######           ####           ##############
########          #########      ####      #######           ####          ###############
##########################################################################################
##########################################################################################


def download(input_user={}): 
    ''' download continuous noise data day per day  for a single year of data. 
    INPUT : 
    -- mode = 0 : initial download. Do not require _metadata directory (containing metadata)
    -- mode = 1 : add : attempt to download ALL missing data in the h5 file. Do not require _metadata
    -- mode = 2 : complete : re-download all 1) new stations, and data that were cancelled 1) because the 
                  datacenter could not be reached or because of a timeout while dowloading/processing the data 
    -- mode = 3 : complete : same as mode 2 but also try to get data that were rejected bc there were incorrect
    -- path : path to the data containing the db.mat file (i.e ../data_5hz/daily/EU/2011/)
    
    OUTPUT : 
    - a single h5 file per day of data : ../data_5hz/daily/EU/2011/day_001.h5 
    - _metadata/day_001.mat and _metadata/day_001.pkl : matlab and python friendly file indicating 
      for each day, which data have been dowloaded or not. If not it indicated the reason 
      (data center not reached, timeout, no data, or incorrect data). 
    - while downloading the data : each time a process work on a day of data it creates a day_001.lock file.
      This is a temporary file which is removed when the day has been fully dowloaded. This file is used 
      to organise the multiprocessing. 
    
    - if mode =1,2,3 : the code creates a _day_001.h5 file for each day for the multiprocessing stuffs. These
                       files should be deleted before re-rerunning the code with different options. 
    
    TYPICAL USAGE : 
    1) configure the download (see configure to get more informations): 
    >>m00_get_data.configure(inp)
    
    2) download the data : 
    >> inp['mode']=0 
    >> inp['path']='../data_5hz/daily/EU/2011/'
    >> m00_get_data.downlad(inp)
    if there is no more .lock file it means that everything if finished.  
    if there are some .lock files left whereas no process is running it means that the process were 
    interrupted. the .lock files and their corresponding .h5 file should be deleted and the code re-runned
    
    
    3) complete the data (may require to remove all _day* files) : 
    >>inp['mode']=2 
    >>inp['path']='../data_5hz/daily/EU/2011/'
    >>m00_get_data.downlad(inp)
    
    4) eventually add new stations (may require do remove all _day* files) :  
    >>m0_get_data2.configure(inp)
    >>inp['mode']=2 or 1  
    >>m00_get_data.downlad(inp)
    
    In this case both mode 1 and 2 could be used the difference is that : 
    mode =1 : do not require a _metadata directory 
            : will attempt to dowload ALL missing data in the h5 files, i.e the new station + all the old one
            :  that failed, whatever the cause was. 
    mode =2 : require a _metadata directory 
            : will attempt to dowload only new stations + previous stations that failed because the datacenter
            : was not reachable or because of a time out (long download/processing time)
    '''
    in_         = {} 
    in_['path'] = './data_test/daily/tag/yyyy/'
    in_['mode'] = 0  # 0=all which has not yet been downloaded 1=timeout 2=timeout+incorrect  
    in_         = lang.parse_options(in_,input_user)
    try :
        db          = pickle.load(open(in_['path']+'/db.pkl', "rb" ))
    except : 
        dd.dispc('You need to configure first !','r','b')
        return
    list_of_dates = np.arange(int(db['in_']['day1'].timestamp),int(db['in_']['day2'].timestamp+86400),86400)
    random.shuffle(list_of_dates)   
    for idate in list_of_dates :
        cday = UTCDateTime(idate)
        path = define_path(in_,cday)
        if has_this_day_already_been_processed(in_['mode'],path) : continue 
        create_lock_file(path['h5_lock_file'])
        md   = load_metadata(path['metadata_pkl'])
        md   = core_download(in_,db,md,path,cday,cday+86400.0)        
        #exiting this day  : write metadata + remove lock file + create a _day_001.done file
        try :
            #h5py bug : can crash when appending an existing file :
            write_metadata_hdf5(path['h5_file'],db['in_'],cday,sta=db['sta'])
        except  : 
            pass
        filename = path['metadata_pkl']
        ff       = open(path['metadata_pkl'],'wb')
        pickle.dump(md,ff)
        ff.close()
        io.savemat(path['metadata_mat'],md)
        os.remove(path['h5_lock_file'])
        if in_['mode'] !=0 :  #pour la gestion du multicore on est oblige de creer un ficher factice
            create_lock_file(path['h5_finished'])
    # exiting this download session :
    if in_['mode'] >0 : # if we generated *_done_files => rm them if there is not *cplt file (code weakness)
        delete_done_files_if_no_more_cplt(in_['path']) 


def download_events(input_user={}): 
    ''' download events . 
    '''
    in_         = {} 
    in_['path'] = './data_test/events/tag/'
    in_['mode'] = 0  # 0=all which has not yet been downloaded 1=timeout 2=timeout+incorrect  
    in_         = lang.parse_options(in_,input_user)
    try :
        db          = pickle.load(open(in_['path']+'/db.pkl', "rb" ))
    except : 
        dd.dispc('You need to configure first !','r','b')
        return
    
    for iev in db['ev']:
        date1 = UTCDateTime(db['ev'][iev]['date'])
        if db['in_']['ev']['time_after']:
            date2 = date1 + db['in_']['ev']['time_after']
        else:
            date2 = date1 + length_events(db['ev'][iev]['mag'])
        if db['in_']['ev']['time_before']:
            date1 = date1 - db['in_']['ev']['time_before']
        path = define_path_evts(in_,db['ev'][iev]['ev_id'])
        if has_this_day_already_been_processed(in_['mode'],path) : continue 
        create_lock_file(path['h5_lock_file'])
        md   = load_metadata(path['metadata_pkl'])
        md   = core_download(in_,db,md,path,date1,date2)        
        #exiting this day  : write metadata + remove lock file + create a _day_001.done file
        try :
            write_metadata_hdf5(path['h5_file'],db['in_'],date1,sta=db['sta'],ev=db['ev'][iev])
        except  : 
            pass
        filename = path['metadata_pkl']
        ff       = open(path['metadata_pkl'],'wb')
        pickle.dump(md,ff)
        ff.close()
        io.savemat(path['metadata_mat'],md)
        os.remove(path['h5_lock_file'])
        if in_['mode'] !=0 :  #pour la gestion du multicore on est oblige de creer un ficher factice
            create_lock_file(path['h5_finished'])
    # exiting this download session :
    if in_['mode'] >0 : # if we generated *_done_files => rm them if there is not *cplt file (code weakness)
        delete_done_files_if_no_more_cplt(in_['path']) 

def length_events(mag):
    return 86400. * (np.exp(mag**3 /450.) -1) 

def core_download(in_,db,md,path,date1,date2): 
    ''' 
    loop over station list and channel for a given range of dates...
    '''
    for ista in db['sta'].values() :
        ista_arg = ista.copy()
        for icmp in db['in_']['cmp'] :
            t0        = time.time()
            queue     = multiprocessing.Queue()                                                
            queue.put([ista_arg,0.])               # the second arg will contain the result 
            time.sleep(0.01)
            
            if has_this_ch_already_been_dowloaded(in_['mode'],path['h5_file'],ista,icmp,md) : continue #
            data_center = ista['dc']
            
            if ista['kname'] not in md  : 
                md[ista['kname']]={}
            md[ista['kname']][icmp] = initialize_metadata()
            ###################
            # from SDS ... same md structure 
            ###################
            if data_center == 'SDS':
                okSDS = True 
                try:
                    if in_['ev']['only_event']:
                        okSDS = False 
                except:
                    pass
                if okSDS:
                    if os.path.isdir(db['in_']['sds_path']) == False : 
                        md[ista['kname']][icmp]['reached_datacenter'] = False 
                        md[ista['kname']][icmp]['output']             = 'could not initialize the client'
                        md[ista['kname']][icmp]['success']            = False 
                        continue
                    #attempt to load the data by running a subprocess that will be killed after X seconds
                    args      = (queue,db['in_'],path['h5_file'],data_center,date1,date2,ista_arg,icmp) 
                else:
                    dd.dispc('Events not implemented for SDS archive yet!','r','b')
            else:
                ###################
                # from client
                ###################
                client = initialize_client(data_center,db['in_'])
                if client == False : 
                    md[ista['kname']][icmp]['reached_datacenter'] = False 
                    md[ista['kname']][icmp]['output']             = 'could not initialize the client'
                    md[ista['kname']][icmp]['success']            = False 
                    continue
                args      = (queue,db['in_'],path['h5_file'],client,date1,date2,ista_arg,icmp)
            #attempt to download the data by running a subprocess that will be killed after X seconds

            ###################
            ###################
            pp        = multiprocessing.Process(target=load_from_source, name="load_from_source", args=args)
            ###################
            ###################

            pp.start()
            pp.join(db['in_']['qc']['timeout'])
            time_out  = False 
            #  did we reach time out ? 
            if pp.is_alive() : 
                dd.dispc('     timeout  after '+str(db['in_']['qc']['timeout'])+'s','r','n')
                pp.terminate()
                pp.join()
                time_out = True
            #get the results :
            result   = queue.get() 
            new_ista = result[0]  # if not timoeout : get updated station info with locid and channel (BH,HH)
            
            db['sta'][ista['kname']] = new_ista.copy()
            ista_arg                 = new_ista.copy()

            if time_out== False :            
                md[ista['kname']][icmp]             = result[1]
                md[ista['kname']][icmp]['time_out'] = False
            else :
                md[ista['kname']][icmp]['output']   = 'time out'
                md[ista['kname']][icmp]['time_out'] = True

            md[ista['kname']][icmp]['sta_info']           = new_ista.copy() 
            md[ista['kname']][icmp]['reached_datacenter'] = True         
            md[ista['kname']][icmp]['time']               = int(time.time()-t0)
            if md[ista['kname']][icmp]['output']=='OK' and md[ista['kname']][icmp]['time_out']==False : 
                md[ista['kname']][icmp]['success'] = True 
            else : 
                md[ista['kname']][icmp]['success'] = False 
            dd.dispc('     '+str(md[ista['kname']][icmp]['time'])+'s','y','n')
    return md







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


#---------------------------------------------------------------------------------------
#-------------------- CONFIGURE SUBFUNCTIONSFUNCTIONS ----------------------------------
#---------------------------------------------------------------------------------------


def read_station_list(filename) :
    sta={}
    ff=open(filename,'r')
    for iline in ff : 
        (dc,net,name,loc,lat,lon,elev,depth) = iline.split()
        if loc=='--' : loc = ''
        if loc == u'' : kname=net+'_'+name+'_00'
        else: kname=net+'_'+name+'_'+loc 
        sta[kname]={}
        sta[kname]['net']  = net
        sta[kname]['name'] = name
        sta[kname]['loc']  = loc
        sta[kname]['kname']= kname
        sta[kname]['lat']  = float(lat)
        sta[kname]['lon']  = float(lon)
        sta[kname]['elev'] = float(elev)
        sta[kname]['depth']= float(depth)
        sta[kname]['dc']   = dc
    ff.close()
    return sta 

#------------------------------------------
def read_event_list(filename) :
    ev={}
    ff=open(filename,'r')
    for iline in ff : 
        (date,lat,lon,depth,mag,mag_type) = iline.split()       
        kname = date+'_'+mag_type+'_'+mag 
        ev[kname]={}
        ev[kname]['date']    = date
        ev[kname]['ev_id']   = kname.replace(':','-')
        ev[kname]['lat']     = float(lat)
        ev[kname]['lon']     = float(lon)
        ev[kname]['depth']   = float(depth)
        ev[kname]['mag']     = float(mag)
        ev[kname]['mag_type']= mag_type
    ff.close()
    return ev 

#------------------------------------------
def write_db_file(out_dir,in_,sta,ev={}) :
    ''' output a db.mat file containing all input parameters of the code as well as the station list
    The db.mat file is in data_5hz/daily/EU/2010 directory. date are converted to matlab format
    '''
    db_file      = out_dir+'/db.mat' 
    db_file_lock = db_file+'.lock'   
    #output db_file in a matlab-friendly way : 
    db                     = dict() 
    db['in_']              = copy.deepcopy(in_)
    db['in_']['day1']      = float(db['in_']['day1'].toordinal()) + 366.
    db['in_']['day2']      = float(db['in_']['day2'].toordinal()) + 366.
    if db['in_']['password'] is None:db['in_']['password']='None'
    if db['in_']['user_id'] is None:db['in_']['user_id']='None'
    if db['in_']['ev']['time_after'] is None:db['in_']['ev']['time_after']='None'
    db['sta']              = dict()
    db['sta']['lon']       = []
    db['sta']['lat']       = []
    db['sta']['sta']       = []
    db['sta']['net']       = []
    db['sta']['dc']        = []
    db['sta']['loc']       = []
    db['sta']['elev']      = []
    db['sta']['depth']     = []
    if ev is not {}:
        db['ev']           = copy.deepcopy(ev)
    for ista in sta :
        db['sta']['sta'].append(ista) 
        db['sta']['net'].append(sta[ista]['net'])
        db['sta']['dc'].append(sta[ista]['dc'])
        db['sta']['lon'].append(float(sta[ista]['lon']))
        db['sta']['lat'].append(float(sta[ista]['lat']))
        db['sta']['loc'].append(sta[ista]['loc'])
        db['sta']['elev'].append(float(sta[ista]['elev']))
        db['sta']['depth'].append(float(sta[ista]['depth']))
    io.savemat(db_file,db)
    #output db file in a python-friendly way :
    db        = dict()
    db['in_'] = in_
    db['sta'] = sta
    if ev is not {}:
        db['ev']           = copy.deepcopy(ev)
    filename  = db_file[0:-4]+'.pkl'
    ff        = open(filename,'wb')
    pickle.dump(db,ff)
    ff.close()


#-----------------------------------------------------------------------------------------------
#----------------------- SINGLE DOWNLOAD AND PROCESSING FUNCTIONS ------------------------------
#-----------------------------------------------------------------------------------------------





def load_from_source(queue,in_,out_file,client,date1,date2,ista,icmp) :
    '''download and decimate/rm instrumental response of a single day/station/cmp and write in a h5 file
    '''
    tap_len     = in_['pp']['tap_len']
    remove_resp = in_['pp']['remove_resp']
    st      = obspy.Stream()
    if isinstance(in_['ch'],list):
        if len(in_['ch']) == 1:
            ista['ch'] = in_['ch'][0]
    elif not isinstance(in_['ch'],list):
        ista['ch'] = in_['ch']

    try :
        if client == 'SDS':
            archive_path = in_['sds_path']
            meta_path    = in_['sds_metapath']
            if 'ch' not in ista :
                dd.dispc('      attempting to find ch (SDS)','c','d')
                st,ista = determine_ch_and_locid_from_archive(ista,icmp,in_['ch'],date1-tap_len,date2+tap_len,archive_path,meta_path,tap_len,remove_resp) 
            else : 
                st = read_SDS(ista['net'],ista['name'],ista['loc'],ista['ch']+icmp,date1-tap_len,date2+tap_len,archive_path,meta_path,tap_len,remove_resp)
        else :
            if 'ch' not in ista :
                dd.dispc('      attempting to find ch (WEB)','c','d')
                st,ista = determine_ch_and_locid_from_client(ista,icmp,in_['ch'],client,date1-tap_len,date2+tap_len)
            else :
                try :
                    st = client.get_waveforms(ista['net'],ista['name'],ista['loc'],ista['ch']+icmp,date1-tap_len,date2+tap_len,attach_response=True)
                except :
                    dd.dispc('      new attempt to find ch (WEB)','c','d')
                    st,ista = determine_ch_and_locid_from_client(ista,icmp,in_['ch'],client,date1-tap_len,date2+tap_len) 
    except :
        dd.dispc('      weird error during (down)load ??!','r','n')

    md = check_if_trace_correct(st,in_['qc']) #faiblesse du code les md doivent synchro ici
    if md['everything_correct'] :
        try :
            st = process_trace(st,in_['pp'],date1,date2)
            if len(st):
                md['correct_process'] = True
        except :
            dd.dispc('      weird error during preprocess ??!','r','n')
            st  = obspy.Stream()
            #md = check_if_trace_correct(st,in_['qc'])
    if md['everything_correct']  == False or md['correct_process'] == False:
        queue.get()
        queue.put([ista,md])
        return
    if in_['format']=='h5':
        write_as_hdf5(out_file,st,ista,icmp)  
        queue.get()
        queue.put([ista,md])


#----------------------------------------
def write_as_hdf5(out_file,st,ista,icmp) :
    loc = st[0].stats.location
    if loc == u'':
        loc  = '00' 
    if icmp=='1':
        dd.dispc('      renaming cmp 1 to N','c','d')
        icmp='N'
    if icmp=='2':
        dd.dispc('      renaming cmp 2 to E','c','d')
        icmp='E'
    dset_name = "/" + ista['net']+"/"+ista['name']+"."+loc+"/"+icmp
    trace = np.float32(st[0].data)
    fout = h5.File(out_file, "a") #on le sort de la boucle ...
    if dset_name in fout : # should not happen normally except if the coda hb interrupted
        del fout[dset_name]
    dset = fout.create_dataset(dset_name, data=trace,chunks=True,compression="gzip", compression_opts=9)
    fout.close()


#----------------------------------------
def write_metadata_hdf5(out_file,in_,cday,sta={},ev={}) :
    fout = h5.File(out_file, "a") #on le sort de la boucle ...
    if not '/_methadata/_00_download/fe' in fout : 
        fout.create_dataset('/_metadata/_00_download/fe', data=in_['pp']['freq'])
        fout.create_dataset('/_metadata/_00_download/t0_UNIX_timestamp', data=cday.timestamp)
    if not '/_metadata/fe' in fout :
        fout.create_dataset('/_metadata/fe', data=in_['pp']['freq'])
    if not '/_metadata/t0_UNIX_timestamp' in fout :
        fout.create_dataset('/_metadata/t0_UNIX_timestamp', data=cday.timestamp)
    if sta is not {}:
        if '/_metadata/sta' not in fout:
            for key in sta:
                for kkey in sta[key]:
                    idsta = sta[key]['net'] + '/' + sta[key]['name'] + '.' + sta[key]['loc'] 
                    if idsta in fout:
                        fout.create_dataset('/_metadata/sta/' + idsta  + '/' +kkey, data=sta[key][kkey])  
    if ev is not {}:
        if '/_metadata/ev' not in fout:
            for key in ev:
                fout.create_dataset('/_metadata/ev/' + key, data=ev[key]) 
    fout.close() 

#-----------------------------------------
def check_if_trace_correct(st,qc)  :
    md = initialize_metadata()
    md['output']               = 'no data' # 'data found but incorrect trace' or 'OK'
    md['nsegment']             = len(st) 
   
    #check that there is some data. return default value otherwise : 
    if md['nsegment']==0 : 
        dd.dispc('      no data found => exiting','r','n')
        return md
    
    # compute total duration of the trace : 
    npts = 0
    sampling_rate = st[0].stats['sampling_rate'] 
    for ist in st : 
        npts = npts + ist.stats['npts'] 
    md['duration'] = npts/float(sampling_rate) 
    dd.dispc('      found '+str(len(st))+ ' segments and a duration of '+str(md['duration'])+'s','c','d')
    #check that the number of segments is ok :
    if md['nsegment']> qc['max_nsegments'] :
        str_=    '      more than '
        dd.dispc(str_+str(qc['max_nsegments'])+' segments ['+str(len(st))+']! => exiting','r','n') 
    else :
        md['correct_nsegment']        = True

    #nan ??
    for ist in st :
        if np.isnan(ist.data).any() :
            dd.dispc('NaN detected => exiting','r','n')
            md['nan_detected']         = True   

    #check that each segment has the same sampling rate : 
    for ist in st : 
        if ist.stats['sampling_rate'] != sampling_rate :
            str_  = '      wrong sampling rate ['
            dd.dispc(str_+str(ist.stats['sampling_rate'])+'] vs '+str(sampling_rate)+' => exiting','r','n')
            md['correct_sampling_rate']=False
        
    #check total duration of the signal is ok :
    if md['duration'] < qc['min_duration'] :
        str_=   '      total_duration less than '
        dd.dispc(str_+str(qc['min_duration']) +' ('+str(int(md['duration']))+') => exiting','r','n')
    else : 
        md['correct_duration']=True
        
    #conclusion : check if everything ok : 
    if md['correct_sampling_rate'] and md['correct_nsegment'] and md['correct_duration'] and not md['nan_detected'] :
        md['everything_correct']   = True 
        md['output']               = 'OK' #
        dd.dispc('      trace is correct, lets process it','g','n')
    else : 
        md['output']               = 'data found but incorrect trace' #
    return md 
    #check that there is no only zeros in a trace 
        
#------------------------------------------
def initialize_metadata() : #used by check_if_trace_correct AND DOWNLOAD 
    md={}  #dictionnaty that will contain infor on the trace  & download => will go into _metadata
    md['output']               = ''    # 'data found but incorrect trace' or 'OK'
    md['duration']             = -1
    md['nsegment']             = -1 
    md['correct_sampling_rate']= True
    md['correct_nsegment']     = False
    md['correct_duration']     = False
    md['correct_process']      = False 
    md['nan_detected']         = False     
    md['everything_correct']   = False
    return md



#----------------------------------------------------------------------
def determine_ch_and_locid_from_client(ista,compo,channel_list,client,date1,date2) :
    # test only ch since locid is given in the station file !
    # but locid can still be '*' (=> keep the first one !)
    st = obspy.Stream()
    for locid in [ista['loc']]: # useless ... but just in case we want to add some
        for ich in channel_list : #BH puis HH si neccessaire 
            ccmp= ich + compo
            try  :
                st = client.get_waveforms(ista['net'],ista['name'],locid,ccmp,date1,date2,attach_response=True)
                break
            except : pass
        if len(st) > 0 :
            st.sort()
            st          = st.select(location = st[0].stats.location)
            ista['loc'] = st[0].stats.location
            ista['ch']  = ich 
            break 
    return st,ista 

#----------------------------------------------------------------------
def determine_ch_and_locid_from_archive(ista,compo,channel_list,date1,date2,archive_path,meta_path,tap_len,remove_resp) :
    # test only ch since locid is given in the station file !
    # but locid can still be '*' (=> keep the first one !)
    st = obspy.Stream()
    for locid in [ista['loc']]: # useless ... but just in case we want to add some
        for ich in channel_list : #BH puis HH si neccessaire 
            ccmp= ich + compo
            try  :
                st = read_SDS(ista['net'],ista['name'],locid,ccmp,date1,date2,archive_path,meta_path,tap_len,remove_resp)
                if len(st) > 0: break
            except : pass
        if len(st) > 0 :
            st.sort()
            st          = st.select(location = st[0].stats.location)
            ista['loc'] = st[0].stats.location
            ista['ch']  = ich 
            break 
    return st,ista 

#----------------------------------------------------------------------
def read_SDS(net,sta,locid,ccmp,date1,date2,archive_path,meta_path,tap_len,remove_resp) :
    # work only with xml metadata...
    # data : archive_path/year/net/ccmp.D/net.sta.loc.ccmp.D.year.jday
    # metadata : meta_path/net/net.sta.station.response.xml
    st = obspy.Stream()
    jday = "%03d"%(date1+tap_len).julday
    file_name = net + "." + sta + "." + locid + "." + ccmp + ".D." + str((date1+tap_len).year) + "." + jday
    meta_name = net + "." + sta + ".station.response.xml"
    data_path = archive_path + "/" + str((date1+tap_len).year) + "/" + net + "/" + sta + "/" + ccmp + ".D/" + file_name
    md_path   = meta_path + "/" + net + "/" + meta_name 
    if os.path.exists(data_path):
        st  = read(data_path)
        if remove_resp and os.path.exists(md_path):
            inv = read_inventory(md_path)
            st.attach_response(inv)
    return st 


#------------------------------------------------------
def process_trace(st,pp,t1,t2) :
    remove_resp       = pp['remove_resp']
    newfreq           = float(pp['freq'])
    oldfreq           = float(st[0].stats.sampling_rate)
    tap_len           = pp['tap_len'] 
    f_prefilt         = pp['f_prefilt'] 
    glitch_correction = pp['glitch']
    theo_npts         = t2 - t1
    Ltrace            = t2 - t1 + 2 * tap_len
    alreadyresample   = False
    if len(st)>1000:
        dd.dispc('      len(st) > 1000 : custom cat .... VERY HIGH FREQ case','y','n')
        #st.resample(newfreq,no_filter=False)
        #alreadyresample   = True
        stout = Stream()
        st2 = Stream()
        inc = 0
        bornN = np.linspace(0,len(st),97).astype('int')
        for nnn in range(len(bornN)-1):
            #print("nnn")
            for tr in st[bornN[nnn]:bornN[nnn+1]]:
                st2.append(tr)._cleanup()
                inc += 1
            st2.decimate(int(oldfreq/newfreq),no_filter=False)
            stout.append(st2[0])._cleanup()
            st2.clear()
        alreadyresample   = True
        st = stout.copy()
        st.merge(method=1,fill_value=0)
        stout.clear()
    else:
        st._cleanup()
        st = manage_gaps(st,f_prefilt,Ltrace)
    if len(st):
        st.detrend(type='constant')
        st.detrend(type='linear')
        if not alreadyresample:
            #st.filter('lowpass',freq=f_prefilt[1],zerophase=True)
            st.filter('lowpass_cheby_2',freq=f_prefilt[1])
    L= 0.0
    for tr in st: 
        if glitch_correction == True :
            dd.dispc('    removing glitch','r','n')
            tr.data = glitchCorrectionWithFactorStd(tr.data, 20, NumberOfStd = 1, FactorReplaceWithStd = 0)
        if not alreadyresample:
            tr = MY_resample(tr, newfreq) 
        L  = L + float(len(tr))
    if L/float((newfreq*Ltrace)) > 0.999:
        st.merge(method=1,fill_value='interpolate')        
        st.taper(100,max_length=600)
        if remove_resp:
            st.remove_response(output="VEL",taper = False ,water_level = 60.0)
            st.taper(100,max_length=600)
        st.filter('bandpass',freqmin=f_prefilt[0],freqmax=f_prefilt[1],zerophase=True) 
        st.trim(starttime=t1-tap_len,endtime=t2+tap_len,fill_value=0,pad=True)
        st.interpolate(newfreq,starttime=t1,npts=int(theo_npts*newfreq),method='cubic')
    elif L/float((newfreq*Ltrace)) > 0.01:
        st.taper(100,max_length=600)
        if remove_resp:
            st.remove_response(output="VEL",taper = False, water_level = 60.0)
            st.taper(100,max_length=600)
        st.filter('bandpass',freqmin=f_prefilt[0],freqmax=f_prefilt[1],zerophase=True) 
        st.merge(method=1,fill_value=0)
        st.trim(starttime=t1-tap_len,endtime=t2+tap_len,fill_value=0,pad=True)
        st.interpolate(newfreq,starttime=t1,npts=int(theo_npts*newfreq),method='cubic')
    else:
        dd.dispc('                                    ... Almost no data','r','blink')
        st.clear()
    return st 




#------------------------------------
def manage_gaps(st,f_prefilt,Ltrace):
    Lgaps=0
    #for listOfgGap in st.get_gaps():
    #    Lgaps = Lgaps+listOfgGap[6]
    #if Lgaps < 3*Ltrace/4 and len(st.get_gaps()) < 24 : # stats sur tous les gaps ...
    st2  = st.copy()
    ig   = 0
    while len(st2.get_gaps()):
        if ig > len(st2.get_gaps())-1:
            break
        conditionGAP1  = st2.get_gaps()[ig][6] < 1/float(f_prefilt[0])
        conditionGAP2  = st2.get_gaps()[ig][6] < 2/float(f_prefilt[0])
        conditionLEN1  = len(st2[ig])   < (4 / float(f_prefilt[0]) * st2[ig].stats.sampling_rate)
        conditionLEN2  = len(st2[ig+1]) < (4 / float(f_prefilt[0]) * st2[ig+1].stats.sampling_rate)    
        condition      = conditionGAP1 or ((conditionLEN1 or conditionLEN2) and conditionGAP2)
        if condition:
            st3 = Stream(traces=[st2[ig], st2[ig+1]])
            st3.merge(method=1,fill_value='interpolate') ## POSSIBLE SMALL PHASE SHIFT ...
            st2.remove(st2[ig+1])
            st2.remove(st2[ig]) 
            st2 = st2+st3
            st2.sort()
            st3.clear()
        else:
            ig = ig+1
    st2._cleanup()
    st = st2.copy()
    st2.clear()
    for tr in st:
        if len(tr) < (4/float(f_prefilt[0])) * tr.stats.sampling_rate:
            st.remove(tr) 
    #else:
    #    dd.dispc('                                    ... gapssss','r','blink')
    #    st.clear()
    return st

#------------------------------------------
def MY_resample(tr, newfrequence):#destroy the trace,
    rateFreq = float(tr.stats.sampling_rate)/float(newfrequence)
    if int(rateFreq)>1:
        if tr.stats.sampling_rate==int(tr.stats.sampling_rate) and rateFreq==int(rateFreq):
            tr.decimate(int(rateFreq),no_filter=True)
        else:
            rateFreq_closest       = round(rateFreq)
            temp_freq              = float(newfrequence) * float(rateFreq_closest)
            tr.interpolate(temp_freq,method='cubic')
            new_rateFreq           = int(float(tr.stats.sampling_rate)/float(newfrequence))
            tr.decimate(new_rateFreq,no_filter=True)
    elif int(rateFreq)<1:
        tr.interpolate(float(newfrequence),method='cubic')
    return tr

def glitchCorrectionWithFactorStd(Trace, FactorTestStd, NumberOfStd = 1, FactorReplaceWithStd = 0):
    for i in range(NumberOfStd):
        arrayReplace = np.ones(len(Trace), dtype ='float')*np.std(Trace)*FactorReplaceWithStd
        arrayReplace *= np.sign(Trace)
        Trace = scipy.where(scipy.absolute(Trace)>FactorTestStd*np.std(Trace), arrayReplace, Trace)
    return Trace


#------------------------------------------------------------------
#------ THE FOLLOWING FUNCTIONS ARE PRIVATE -----------------------
#------ DOWNLOAD SUBFUNCTIONS  ------------------------------------
#------------------------------------------------------------------


def initialize_client(data_center,db) :
    try :
        if data_center=='IRISPH5':
            dd.dispc('  IRISPH5 ...','b','n')
            DATASELECT="http://service.iris.edu/ph5ws/dataselect/1"
            client = obspy.clients.fdsn.Client(service_mappings={'dataselect': DATASELECT})
        else:
            client = obspy.clients.fdsn.Client(data_center,user=db['user_id'],password=db['password'])
    except : 
        client = False
        dd.dispc('  data center ' + data_center + ' cannot be reach','r','n')
    return client 
                        

#------------------------------------------
def has_this_ch_already_been_dowloaded(mode,h5_filename,ista,icmp,md) : 
    if mode == 0 : 
        done = False 
    elif mode ==1 :  #check in the h5 file if it is here or not 
        if os.path.isfile(h5_filename) == False :
            done = False
        else :
            f_h5 =h5.File(h5_filename, "r")
            if 'loc' in ista and len(ista['loc'])>0:
                if icmp=='1':
                    path_in_h5='/'+ista['net']+'/'+ista['name']+'.'+ista['loc']+'/N'
                elif icmp=='2':
                    path_in_h5='/'+ista['net']+'/'+ista['name']+'.'+ista['loc']+'/E'
                else:
                    path_in_h5='/'+ista['net']+'/'+ista['name']+'.'+ista['loc']+'/'+icmp
            else:
                if icmp=='1':
                    path_in_h5='/'+ista['net']+'/'+ista['name']+'.00/N'
                elif icmp=='2':
                    path_in_h5='/'+ista['net']+'/'+ista['name']+'.00/E'
                else:
                    path_in_h5='/'+ista['net']+'/'+ista['name']+'.' + '00/'+icmp
            if path_in_h5 in f_h5 : 
                done = True 
            else : 
                done = False 
            f_h5.close()
    else : 
        done = True
        output=''
        data_center_reachable = False
        it_is_a_new_station = True 
        if ista['kname'] in md  :                           # station is in the metadata
            if icmp in md[ista['kname']] :                  # station/channel is in the metadata
                it_is_new_station = False                   # => it is not a new station/ch 
                if 'output' in md[ista['kname']][icmp]:       # should not be useful
                    output=md[ista['kname']][icmp]['output']
                    data_center_reachable= md[ista['kname']][icmp]['reached_datacenter'] 
                else :
                    data_center_reachable= False             #should never happen actually

        if output== 0. or output=='':          # can happend if the code was interrupted by a ctrl +C 
            output ='    == 0.0 CTRL-C?'
            data_center_reachable = False      # we force the download 

        if mode ==2 :
            if data_center_reachable == False : done = False
            if data_center_reachable == True and output=='time out' : done = False
     
        if mode ==3 : 
            dd.dispc('mode=3','r','b')
            if data_center_reachable == False : done = False
            if data_center_reachable == True and output=='time out' : done = False
            if data_center_reachable == True and output=='data found but incorrect trace' : done = False
        
            
        if it_is_a_new_station == True : 
            dd.dispc('    it is a new station/ch  => we try to download it','y','n')
        if data_center_reachable == False : 
            dd.dispc('    the data center was not reached previously => we retry','y','n')
        if len(output) > 0 and done ==False :
            dd.dispc('    output = '+output+' => we try to download it','y','n')
        if len(output) > 0 and done ==True :
            dd.dispc('    output = '+output+' => we do not try to download it','y','n')

    sta_id=ista['net']+'.'+ista['name']+'.'+icmp
    if done == True  : dd.dispc('     '+sta_id+' already done : do not attempt to get it','r','n')
    if done == False : dd.dispc('     '+sta_id+' downloading...','c','b')
    return done 


#------------------------------------------
def load_metadata(filename) : 
    ''' load existing metadata file for the current day, or initialize it'''
    if os.path.isfile(filename) : 
        md=pickle.load( open(filename, "rb" ) )
    else :
        md={}
    return md 


#------------------------------------------
def has_this_day_already_been_processed(mode,path) :
    if mode == 0 : # download from scratch. Just check if a .h5 or .lock file exist :
        if os.path.isfile(path['h5_file']) | os.path.isfile(path['h5_lock_file']) :
            done = True 
        else : 
            done = False 
    else :  # complete the data set : 
        if os.path.isfile(path['h5_finished']) | os.path.isfile(path['h5_lock_file']) :
            done = True 
        else : 
            done = False 
    if done == True  : dd.dispc('  '+path['h5_file'] + ' is already (being) dowloaded','r','n')
    if done == False : dd.dispc('  '+path['h5_file'] + ': lets go','y','b')
    return done 


#------------------------------------------
def delete_done_files_if_no_more_cplt(in_path) :
    cplt_files=glob.glob(in_path+'/*cplt')     # we reached the last day and mode >0
    if len(cplt_files) > 0 :                   # and there is no more cplt files
        return 
    lock_file=in_path+'/_cleaning.lock'        # so we delete all _*done files 
    if os.path.isfile(lock_file) :             # no one else is cleaning 
        return 
    dd.dispc('  FINISHED ! => removing all _*done files','y','b')
    create_lock_file(lock_file)                # so lets go 
    done_files=glob.glob(in_path+'/_*done')    # 
    for ifile in done_files :                  
        os.remove(ifile) 
    os.remove(lock_file)


#------------------------------------------
def define_path(in_,cday) : 
    path                = {}
    path['out_dir']     = in_['path']
    path['h5_file']     = path['out_dir']+'/day_'+str(cday.julday).zfill(3)+'.h5'
    path['h5_finished'] = path['out_dir']+'/_day_'+str(cday.julday).zfill(3)+'.done'
    if in_['mode']==0 :
        path['h5_lock_file']  = path['h5_file']+'.lock'
    else :
        path['h5_lock_file']  = path['h5_file']+'.cplt'
    path['metadata_dir']  = path['out_dir']+'/_metadata'
    path['metadata_mat']  = path['metadata_dir']+'/'+str(cday.julday).zfill(3)+'.mat'
    path['metadata_pkl']  = path['metadata_dir']+'/'+str(cday.julday).zfill(3)+'.pkl'
    mkdir(path['metadata_dir'])
    return path 

#------------------------------------------
def define_path_evts(in_,ev_id) : 
    path                = {}
    path['out_dir']     = in_['path']
    path['h5_file']     = path['out_dir']+'/'+ev_id+'.h5'
    path['h5_finished'] = path['out_dir']+'/_'+ev_id+'.done'
    if in_['mode']==0 :
        path['h5_lock_file']  = path['h5_file']+'.lock'
    else :
        path['h5_lock_file']  = path['h5_file']+'.cplt'
    path['metadata_dir']  = path['out_dir']+'/_metadata'
    path['metadata_mat']  = path['metadata_dir']+'/'+ev_id+'.mat'
    path['metadata_pkl']  = path['metadata_dir']+'/'+ev_id+'.pkl'
    mkdir(path['metadata_dir'])
    return path 

#------------------------------------------
def create_lock_file(filename) : 
    ff=open(filename,'wb')
    pickle.dump(filename,ff)
    ff.close()





#--------------------- subfunctions common to configure and dowload : 


def mkdir(out_dir) : 
    if os.path.isdir(out_dir) == False :
        os.makedirs(out_dir)


#------------------------------------------
#------------------------------------------
#------------------------------------------
#------------------------------------------
def get_network_table() : 
    # updated on May 3 2017
    #dd.dispc('WARNING : known bug for temporary arrays in get_network_table: net id could exists on multiple DC... ','c','blink')
    table={}
    table['GR']='BGR'; table['SX']='BGR'; table['TH']='BGR'; table['8X']='ETH'; 
    table['CH']='ETH'; table['S']='ETH'; table['XH']='ETH'; table['XT']='ETH'; 
    table['Z3']='ETH'; table['1B']='GFZ'; table['1G']='GFZ'; table['2B']='GFZ'; 
    table['2D']='GFZ'; table['2F']='GFZ'; table['2G']='GFZ'; table['3D']='GFZ'; 
    table['3E']='GFZ'; table['3H']='GFZ'; table['4A']='GFZ'; table['4B']='GFZ'; 
    table['4C']='GFZ'; table['5C']='GFZ'; table['5E']='GFZ'; table['6A']='GFZ'; 
    table['6C']='GFZ'; table['6E']='GFZ'; table['7A']='GFZ'; table['7B']='GFZ'; 
    table['7E']='GFZ'; table['7G']='GFZ'; table['8A']='GFZ'; table['8E']='GFZ'; 
    table['8F']='GFZ'; table['9A']='GFZ'; table['9C']='GFZ'; table['AF']='GFZ'; 
    table['AW']='GFZ'; table['CK']='GFZ'; table['CN']='GFZ'; table['CX']='GFZ'; 
    table['CZ']='GFZ'; table['DK']='GFZ'; table['EE']='GFZ'; table['EI']='GFZ'; 
    table['FN']='GFZ'; table['GE']='GFZ'; table['HE']='GFZ'; table['HT']='GFZ'; 
    table['HU']='GFZ'; table['IA']='GFZ'; table['IQ']='GFZ'; table['IS']='GFZ'; 
    table['JS']='GFZ'; table['KC']='GFZ'; table['KV']='GFZ'; table['NU']='GFZ'; 
    table['PL']='GFZ'; table['PM']='GFZ'; table['PZ']='GFZ'; table['SJ']='GFZ'; 
    table['SK']='GFZ'; table['TT']='GFZ'; table['UP']='GFZ'; table['WM']='GFZ'; 
    table['X1']='GFZ'; table['X6']='GFZ'; table['XC']='GFZ'; table['XE']='GFZ'; 
    table['XF']='GFZ'; table['XN']='GFZ'; table['XO']='GFZ'; table['XP']='GFZ'; 
    table['Y4']='GFZ'; table['Y7']='GFZ'; table['Y9']='GFZ'; table['YU']='GFZ'; 
    table['Z2']='GFZ'; table['Z3']='GFZ'; table['Z4']='GFZ'; table['Z5']='GFZ'; 
    table['Z6']='GFZ'; table['ZA']='GFZ'; table['ZB']='GFZ'; table['ZC']='GFZ'; 
    table['ZD']='GFZ'; table['ZE']='GFZ'; table['ZF']='GFZ'; table['ZG']='GFZ'; 
    table['ZO']='GFZ'; table['ZP']='GFZ'; table['ZQ']='GFZ'; table['ZR']='GFZ'; 
    table['ZS']='GFZ'; table['ZV']='GFZ'; table['ZW']='GFZ'; table['ZZ']='GFZ'; 
    table['3A']='INGV'; table['4A']='INGV'; table['4C']='INGV'; table['AC']='INGV'; 
    table['BA']='INGV'; table['GU']='INGV'; table['IV']='INGV'; table['IX']='INGV'; 
    table['MN']='INGV'; table['NI']='INGV'; table['OT']='INGV'; table['OX']='INGV'; 
    table['RF']='INGV'; table['SI']='INGV'; table['ST']='INGV'; table['TV']='INGV'; 
    table['XO']='INGV'; table['Z3']='INGV'; table['G']='IPGP'; table['GL']='IPGP'; 
    table['MQ']='IPGP'; table['PF']='IPGP'; table['WI']='IPGP'; table['1A']='IRIS'; 
    table['1C']='IRIS'; table['1D']='IRIS'; table['1E']='IRIS'; table['1F']='IRIS'; 
    table['2A']='IRIS'; table['2B']='IRIS'; table['2C']='IRIS'; table['2D']='IRIS'; 
    table['2G']='IRIS'; table['2H']='IRIS'; table['3A']='IRIS'; table['3C']='IRIS'; 
    table['3D']='IRIS'; table['3E']='IRIS'; table['3F']='IRIS'; table['4A']='IRIS'; 
    table['4E']='IRIS'; table['4F']='IRIS'; table['5A']='IRIS'; table['5C']='IRIS'; 
    table['5E']='IRIS'; table['5F']='IRIS'; table['6C']='IRIS'; table['6D']='IRIS'; 
    table['6E']='IRIS'; table['6F']='IRIS'; table['7A']='IRIS'; table['7B']='IRIS'; 
    table['7C']='IRIS'; table['7D']='IRIS'; table['7E']='IRIS'; table['7F']='IRIS'; 
    table['7G']='IRIS'; table['7J']='IRIS'; table['8A']='IRIS'; table['8B']='IRIS'; 
    table['8E']='IRIS'; table['8G']='IRIS'; table['9B']='IRIS'; table['9D']='IRIS'; 
    table['9F']='IRIS'; table['9G']='IRIS'; table['AC']='IRIS'; table['AE']='IRIS'; 
    table['AF']='IRIS'; table['AG']='IRIS'; table['AI']='IRIS'; table['AK']='IRIS'; 
    table['AO']='IRIS'; table['AR']='IRIS'; table['AS']='IRIS'; table['AT']='IRIS'; 
    table['AU']='IRIS'; table['AV']='IRIS'; table['AX']='IRIS'; table['AY']='IRIS'; 
    table['AZ']='IRIS'; table['BC']='IRIS'; table['BE']='IRIS'; table['BF']='IRIS'; 
    table['BI']='IRIS'; table['BL']='IRIS'; table['C']='IRIS'; table['C0']='IRIS'; 
    table['C1']='IRIS'; table['CA']='IRIS'; table['CB']='IRIS'; table['CC']='IRIS'; 
    table['CD']='IRIS'; table['CH']='IRIS'; table['CI']='IRIS'; table['CK']='IRIS'; 
    table['CM']='IRIS'; table['CN']='IRIS'; table['CO']='IRIS'; table['CS']='IRIS'; 
    table['CT']='IRIS'; table['CU']='IRIS'; table['CW']='IRIS'; table['CY']='IRIS'; 
    table['CZ']='IRIS'; table['DK']='IRIS'; table['DR']='IRIS'; table['DW']='IRIS'; 
    table['EC']='IRIS'; table['EI']='IRIS'; table['EP']='IRIS'; table['ER']='IRIS'; 
    table['ET']='IRIS'; table['FR']='IRIS'; table['GB']='IRIS'; table['GE']='IRIS'; 
    table['GH']='IRIS'; table['GI']='IRIS'; table['GL']='IRIS'; table['GO']='IRIS'; 
    table['GS']='IRIS'; table['GT']='IRIS'; table['GY']='IRIS'; table['H2']='IRIS'; 
    table['HG']='IRIS'; table['HK']='IRIS'; table['HL']='IRIS'; table['HT']='IRIS'; 
    table['HV']='IRIS'; table['HW']='IRIS'; table['IC']='IRIS'; table['IE']='IRIS'; 
    table['II']='IRIS'; table['IM']='IRIS'; table['IN']='IRIS'; table['IO']='IRIS'; 
    table['IP']='IRIS'; table['IU']='IRIS'; table['IW']='IRIS'; table['JM']='IRIS'; 
    table['JP']='IRIS'; table['KC']='IRIS'; table['KG']='IRIS'; table['KN']='IRIS'; 
    table['KO']='IRIS'; table['KP']='IRIS'; table['KR']='IRIS'; table['KS']='IRIS'; 
    table['KW']='IRIS'; table['KY']='IRIS'; table['KZ']='IRIS'; table['LB']='IRIS'; 
    table['LD']='IRIS'; table['LI']='IRIS'; table['LO']='IRIS'; table['LX']='IRIS'; 
    table['MB']='IRIS'; table['MC']='IRIS'; table['MG']='IRIS'; table['MI']='IRIS'; 
    table['MM']='IRIS'; table['MN']='IRIS'; table['MQ']='IRIS'; table['MR']='IRIS'; 
    table['MS']='IRIS'; table['MX']='IRIS'; table['MY']='IRIS'; table['N4']='IRIS'; 
    table['NA']='IRIS'; table['NB']='IRIS'; table['ND']='IRIS'; table['NE']='IRIS'; 
    table['NI']='IRIS'; table['NJ']='IRIS'; table['NL']='IRIS'; table['NM']='IRIS'; 
    table['NN']='IRIS'; table['NO']='IRIS'; table['NP']='IRIS'; table['NR']='IRIS'; 
    table['NU']='IRIS'; table['NV']='IRIS'; table['NW']='IRIS'; table['NX']='IRIS'; 
    table['NY']='IRIS'; table['NZ']='IRIS'; table['OE']='IRIS'; table['OK']='IRIS'; 
    table['ON']='IRIS'; table['OO']='IRIS'; table['OV']='IRIS'; table['OX']='IRIS'; 
    table['OZ']='IRIS'; table['PA']='IRIS'; table['PB']='IRIS'; table['PE']='IRIS'; 
    table['PF']='IRIS'; table['PI']='IRIS'; table['PL']='IRIS'; table['PM']='IRIS'; 
    table['PN']='IRIS'; table['PO']='IRIS'; table['PR']='IRIS'; table['PS']='IRIS'; 
    table['PT']='IRIS'; table['PY']='IRIS'; table['RC']='IRIS'; table['RE']='IRIS'; 
    table['RM']='IRIS'; table['RO']='IRIS'; table['RS']='IRIS'; table['RV']='IRIS'; 
    table['S']='IRIS'; table['SB']='IRIS'; table['SC']='IRIS'; table['SE']='IRIS'; 
    table['SH']='IRIS'; table['SN']='IRIS'; table['SP']='IRIS'; table['SR']='IRIS'; 
    table['SS']='IRIS'; table['SV']='IRIS'; table['TA']='IRIS'; table['TC']='IRIS'; 
    table['TD']='IRIS'; table['TJ']='IRIS'; table['TM']='IRIS'; table['TO']='IRIS'; 
    table['TR']='IRIS'; table['TS']='IRIS'; table['TT']='IRIS'; table['TW']='IRIS'; 
    table['TX']='IRIS'; table['UK']='IRIS'; table['UO']='IRIS'; table['US']='IRIS'; 
    table['UU']='IRIS'; table['UW']='IRIS'; table['VE']='IRIS'; table['VU']='IRIS'; 
    table['WA']='IRIS'; table['WC']='IRIS'; table['WI']='IRIS'; table['WM']='IRIS'; 
    table['WU']='IRIS'; table['WY']='IRIS'; table['X1']='IRIS'; table['X2']='IRIS'; 
    table['X3']='IRIS'; table['X4']='IRIS'; table['X5']='IRIS'; table['X6']='IRIS'; 
    table['X7']='IRIS'; table['X8']='IRIS'; table['X9']='IRIS'; table['XA']='IRIS'; 
    table['XB']='IRIS'; table['XC']='IRIS'; table['XD']='IRIS'; table['XE']='IRIS'; 
    table['XF']='IRIS'; table['XG']='IRIS'; table['XH']='IRIS'; table['XI']='IRIS'; 
    table['XJ']='IRIS'; table['XK']='IRIS'; table['XL']='IRIS'; table['XM']='IRIS'; 
    table['XN']='IRIS'; table['XO']='IRIS'; table['XP']='IRIS'; table['XQ']='IRIS'; 
    table['XR']='IRIS'; table['XS']='IRIS'; table['XT']='IRIS'; table['XU']='IRIS'; 
    table['XV']='IRIS'; table['XW']='IRIS'; table['XX']='IRIS'; table['XY']='IRIS'; 
    table['XZ']='IRIS'; table['Y1']='IRIS'; table['Y2']='IRIS'; table['Y3']='IRIS'; 
    table['Y5']='IRIS'; table['Y6']='IRIS'; table['Y7']='IRIS'; table['Y8']='IRIS'; 
    table['Y9']='IRIS'; table['YA']='IRIS'; table['YB']='IRIS'; table['YC']='IRIS'; 
    table['YD']='IRIS'; table['YE']='IRIS'; table['YF']='IRIS'; table['YG']='IRIS'; 
    table['YH']='IRIS'; table['YI']='IRIS'; table['YJ']='IRIS'; table['YK']='IRIS'; 
    table['YL']='IRIS'; table['YM']='IRIS'; table['YN']='IRIS'; table['YO']='IRIS'; 
    table['YP']='IRIS'; table['YQ']='IRIS'; table['YR']='IRIS'; table['YS']='IRIS'; 
    table['YT']='IRIS'; table['YU']='IRIS'; table['YV']='IRIS'; table['YW']='IRIS'; 
    table['YX']='IRIS'; table['YY']='IRIS'; table['YZ']='IRIS'; table['Z1']='IRIS'; 
    table['Z2']='IRIS'; table['Z3']='IRIS'; table['Z4']='IRIS'; table['Z5']='IRIS'; 
    table['Z6']='IRIS'; table['Z8']='IRIS'; table['Z9']='IRIS'; table['ZA']='IRIS'; 
    table['ZB']='IRIS'; table['ZC']='IRIS'; table['ZD']='IRIS'; table['ZE']='IRIS'; 
    table['ZF']='IRIS'; table['ZG']='IRIS'; table['ZH']='IRIS'; table['ZI']='IRIS'; 
    table['ZJ']='IRIS'; table['ZK']='IRIS'; table['ZL']='IRIS'; table['ZM']='IRIS'; 
    table['ZN']='IRIS'; table['ZO']='IRIS'; table['ZP']='IRIS'; table['ZQ']='IRIS'; 
    table['ZR']='IRIS'; table['ZS']='IRIS'; table['ZT']='IRIS'; table['ZU']='IRIS'; 
    table['ZV']='IRIS'; table['ZW']='IRIS'; table['ZX']='IRIS'; table['ZZ']='IRIS'; 
    table['6G']='KOERI'; table['IJ']='KOERI'; table['KO']='KOERI'; table['TL']='KOERI'; 
    table['BW']='LMU'; table['Z3']='LMU'; table['AZ']='NCEDC'; table['BK']='NCEDC'; 
    table['CE']='NCEDC'; table['CI']='NCEDC'; table['GM']='NCEDC'; table['GS']='NCEDC'; 
    table['LA']='NCEDC'; table['NC']='NCEDC'; table['NN']='NCEDC'; table['NP']='NCEDC'; 
    table['PB']='NCEDC'; table['PG']='NCEDC'; table['TA']='NCEDC'; table['UO']='NCEDC'; 
    table['US']='NCEDC'; table['UW']='NCEDC'; table['WR']='NCEDC'; table['BS']='NIEP'; 
    table['MD']='NIEP'; table['RO']='NIEP'; table['CQ']='NOA'; table['HA']='NOA'; 
    table['HC']='NOA'; table['HL']='NOA'; table['HP']='NOA'; table['ME']='NOA'; 
    table['X5']='NOA'; table['AB']='ORFEUS'; table['AI']='ORFEUS'; table['BE']='ORFEUS'; 
    table['BN']='ORFEUS'; table['CA']='ORFEUS'; table['CR']='ORFEUS'; table['DZ']='ORFEUS'; 
    table['EB']='ORFEUS'; table['EC']='ORFEUS'; table['ES']='ORFEUS'; table['GB']='ORFEUS'; 
    table['GO']='ORFEUS'; table['HF']='ORFEUS'; table['IB']='ORFEUS'; table['II']='ORFEUS'; 
    table['IP']='ORFEUS'; table['IU']='ORFEUS'; table['LC']='ORFEUS'; table['LX']='ORFEUS'; 
    table['NA']='ORFEUS'; table['NL']='ORFEUS'; table['NO']='ORFEUS'; table['NR']='ORFEUS'; 
    table['NS']='ORFEUS'; table['OE']='ORFEUS'; table['SL']='ORFEUS'; table['TU']='ORFEUS'; 
    table['UP']='ORFEUS'; table['VI']='ORFEUS'; table['Z3']='ORFEUS'; table['1A']='RESIF'; 
    table['3A']='RESIF'; table['4C']='RESIF'; table['7C']='RESIF'; table['7H']='RESIF'; 
    table['CL']='RESIF'; table['FR']='RESIF'; table['MT']='RESIF'; table['ND']='RESIF'; 
    table['RD']='RESIF'; table['X7']='RESIF'; table['XG']='RESIF'; table['XK']='RESIF'; 
    table['XS']='RESIF'; table['XW']='RESIF'; table['XY']='RESIF'; table['Y2']='RESIF'; 
    table['Y4']='RESIF'; table['Y9']='RESIF'; table['YA']='RESIF'; table['YB']='RESIF'; 
    table['YI']='RESIF'; table['YJ']='RESIF'; table['YO']='RESIF'; table['YP']='RESIF'; 
    table['YR']='RESIF'; table['YT']='RESIF'; table['YV']='RESIF'; table['YX']='RESIF'; 
    table['YZ']='RESIF'; table['Z3']='RESIF'; table['ZF']='RESIF'; table['ZH']='RESIF'; 
    table['ZI']='RESIF'; table['ZO']='RESIF'; table['ZS']='RESIF'; table['ZU']='RESIF'; 
    table['AZ']='SCEDC'; table['BC']='SCEDC'; table['CE']='SCEDC'; table['CI']='SCEDC'; 
    table['MX']='SCEDC'; table['NC']='SCEDC'; table['NN']='SCEDC'; table['NP']='SCEDC'; 
    table['PB']='SCEDC'; table['PG']='SCEDC'; table['RB']='SCEDC'; table['SN']='SCEDC'; 
    table['TA']='SCEDC'; table['WR']='SCEDC'; table['ZY']='SCEDC'; table['BL']='USGS'; 
    table['BR']='USGS'; table['XC']='USGS'; table['Y4']='USGS'; 
    return table 



