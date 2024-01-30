import sys
import os 
import glob
import time
import warnings
import copy
import h5py as h5
#import silx.io as h5silx
import numpy as np 
import scipy.fftpack
import scipy.signal
import scipy.io as io
import skimage.restoration as deconv
from math import cos, sin, radians
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import offset_copy
import cartopy.crs as ccrs
from cartopy.io.img_tiles import Stamen
import cartopy.feature as cfeature
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics import locations2degrees
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import dd
import lang

import cProfile
import ipdb


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
        sta[kname]['lat']  = lat
        sta[kname]['lon']  = lon
        sta[kname]['elev'] = elev
        sta[kname]['depth']= depth
        sta[kname]['dc']   = dc
    ff.close()
    return sta
    

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
    
    
    
def write_h5(filename,data={}):
    basename = filename.split("/")[-1]
    dirpath  = filename.split(basename)[0]
    if dirpath=='' or dirpath=='/':
        dirpath = '.'
    if os.path.isdir(dirpath)==False :
        os.makedirs(dirpath)
    fout = h5.File(filename, "a")
    #h5silx.dictdump.dicttoh5(data, filename, h5path='/', mode='a', overwrite_data=True, create_dataset_args=None)    
    dic_to_dataset(fout,data)
    fout.close()


def dic_to_dataset(fout,dic,lname=''):
    for kname in dic.keys():
        llname = lname
        try:
            if '/' + kname in fout:
                fout.__delitem__('/' + kname)        
            if type(dic[kname])==dict:
                fout.create_group(kname)
                llname = lname + '/' + kname
                dic_to_dataset(fout[kname],dic[kname],llname)
            else:
                if dic[kname] is None:
                    fout.create_dataset(lname + '/' + kname, data='None')
                else:        
                    fout.create_dataset(lname + '/' + kname, data=dic[kname])
        except:
            #ipdb.set_trace()
            warnings.warn("problem saving " + lname + '/' + kname)


def load_h5(filename,dset=''):
    fin = h5.File(filename, "r")
    dic = {}
    if dset=='':
        for kname in fout:
            dic[kname] = fout['/' + kname][()]
    else:
        dic[dset] = fout['/' + dset][()]
    fout.close()
    return dic


def recursively_load_dic_from_h5(fin, path='/'):
    dic = {}
    for kname, item in fin[path].items():
        if isinstance(item, h5._hl.dataset.Dataset):
            dic[kname] = item[()]
            if dic[kname]==b'None':
                dic[kname] = None
        elif isinstance(item, h5._hl.group.Group):
            dic[kname] = recursively_load_dic_from_h5(fin, path + kname + '/')
    return dic

