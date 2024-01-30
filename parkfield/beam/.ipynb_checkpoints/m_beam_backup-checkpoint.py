import sys
import os 
import glob
import time
import warnings
import copy
import time
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

import io_beam
import plt_beam
import dd
import lang


import cProfile
import ipdb
    
################################################################
################################################################
################################################################
################################################################
# BEAM


def beam(data,freq,U,pos,u0=np.matrix([0,0]),fktype='bartlett'):
    t0 = time.time()
    if np.isscalar(freq):
        print("        beamforming freq : %.2f"%freq)
    else:
        print("        beamforming freq : %.2f"%freq[0] + " - %.2f"%freq[-1])
       
    freq = np.matrix(freq)
    u0   = np.matrix(u0).transpose()
    pos  = np.matrix(pos)
    fk   = np.zeros([freq.size,U.size,U.size],dtype=np.complex128)
    arf  = fk.copy()
    U    = U.tolist()
    Nb   = np.shape(pos)[0]      
    k0   = 2 * np.pi * u0 * freq    
    
    phase0   = pos * k0
    amp0     = [np.mean(np.abs(data),axis=0)] * data.shape[0]
    replica0 = amp0 * np.array(np.exp(1j * phase0.T)) #./ norm(exp(1j*np.transpose(phase0))) ;    

    if fktype=='classic' :
        for ux in U:
            for uy in U:
                s        = np.matrix([ux,uy]).transpose()
                k        = 2 * np.pi * s * freq
                phase    = - pos * k
                replica  = np.array(np.exp(1j * phase.transpose())) #./ norm(exp(1j*np.transpose(phase)))
                fk[:,U.index(ux),U.index(uy)]  = np.mean(data  * replica,axis=1)
                arf[:,U.index(ux),U.index(uy)] = np.mean(replica0 * replica,axis=1)        
        
    else:
        K,K0 = cross_spectral_mat(data,replica0,Nb,fktype)

        if fktype=='bartlett' :
            for fi in range(freq.size):
                for ux in U : 
                    for uy in U :
                        s       = np.matrix([ux,uy]).transpose()
                        k       = 2 * np.pi * s * freq[0,fi] 
                        phase   = - pos * k
                        replica = np.exp(1j*phase.transpose()) #/ np.linalg.norm(np.exp(1j*phase))
                        fk[[fi],U.index(ux),U.index(uy)]  = (np.conj(replica) * K[[fi],:,:].squeeze() * replica.T).squeeze()/Nb
                        arf[[fi],U.index(ux),U.index(uy)] = (np.conj(replica) * K0[[fi],:,:].squeeze() * replica.T ).squeeze()/Nb    

        if fktype=='mvdr':
            for fi in range(freq.size):
                Kinvf = np.linalg.inv( K[[fi],:,:].squeeze() )
                K0invf= np.linalg.inv( K0[[fi],:,:].squeeze() )
                for ux in U: 
                    for uy in U : 
                        s       = np.matrix([ux,uy]).transpose()
                        k       = 2 * np.pi * s * freq[0,fi]
                        phase   = - pos * k                 
                        replica = np.exp(1j*phase.transpose()) #/ np.linalg.norm(np.exp(1j*phase))
                        fk[[fi],U.index(ux),U.index(uy)]  = (1/(np.conj(replica) * Kinvf * replica.T)).squeeze()/Nb
                        arf[[fi],U.index(ux),U.index(uy)] = (1/(np.conj(replica) * K0invf * replica.T)).squeeze()/Nb
    
    print("        Done in %.2f sec"%(time.time()-t0))
    return fk,arf


def beam_H(dataE,dataN,freq,U,pos,u0=np.matrix([0,0]),output_compo='T',fktype='bartlett'):
    t0 = time.time()
    if np.isscalar(freq):
        print("        beamforming freq : %.2f"%freq)
    else:
        print("        beamforming freq : %.2f"%freq[0] + " - %.2f"%freq[-1])
       
    freq = np.matrix(freq)
    u0   = np.matrix(u0).transpose()
    pos  = np.matrix(pos)
    fk   = np.zeros([freq.size,U.size,U.size],dtype=np.complex128)
    fk_E = np.zeros([freq.size,U.size,U.size],dtype=np.complex128)
    fk_N = np.zeros([freq.size,U.size,U.size],dtype=np.complex128)
    arf  = fk.copy()
    baz  = np.zeros([U.size,U.size])
    U    = U.tolist()
    Nb   = np.shape(pos)[0]      
    k0   = 2 * np.pi * u0 * freq    
    
    phase0   = pos * k0
    amp0     = [(np.mean(np.abs(dataN),axis=0) + np.mean(np.abs(dataE),axis=0)) / 2] * dataN.shape[0]
    replica0 = amp0 * np.array(np.exp(1j * phase0.T)) #./ norm(exp(1j*np.transpose(phase0))) ;    
    #replica0_E = amp0 * np.array(np.exp(1j * phase0.T)) #./ norm(exp(1j*np.transpose(phase0))) ;    
    #replica0_N = amp0 * np.array(np.exp(1j * phase0.T)) #./ norm(exp(1j*np.transpose(phase0))) ;    
    
    for ux in U:
        for uy in U:
            baz[U.index(ux),U.index(uy)]  = np.pi-np.arctan2(uy,ux)
    
    if fktype=='classic' :
        
        for ux in U:
            for uy in U:
                ba     = baz[U.index(ux),U.index(uy)] 
                if output_compo=='T':
                    dataRT       = - dataE * sin(ba) - dataN * cos(ba)
                    #replica0_RT  = - replica0_E * sin(ba) - replica0_N * cos(ba)
                elif output_compo=='R':
                    dataRT  = - dataE * cos(ba) + dataN * sin(ba)
                    #replica0_RT  = - replica0_E * cos(ba) + replica0_N * sin(ba)
                else:
                    raise ValueError("output_compo = 'T' or 'R'")
                s        = np.matrix([ux,uy]).transpose()
                k        = 2 * np.pi * s * freq
                phase    = - pos * k
                replica  = np.array(np.exp(1j * phase.transpose())) #./ norm(exp(1j*np.transpose(phase)))
                fk[:,U.index(ux),U.index(uy)]  = np.mean(dataRT  * replica,axis=1)                
                arf[:,U.index(ux),U.index(uy)] = np.mean(replica0 * replica,axis=1)        
        
    else:

        if fktype=='bartlett' :
            for fi in range(freq.size):
                for ux in U: 
                    for uy in U:
                        ba     = baz[U.index(ux),U.index(uy)] 
                        if output_compo=='T':
                            dataRT  = - dataE * sin(ba) - dataN * cos(ba)
                        elif output_compo=='R':
                            dataRT  = - dataE * cos(ba) + dataN * sin(ba)
                        else:
                            raise ValueError("output_compo = 'T' or 'R'")
                        K,K0     = cross_spectral_mat(dataRT[[fi],:],replica0[[fi],:],Nb,fktype)   
                        s       = np.matrix([ux,uy]).transpose()
                        k       = 2 * np.pi * s * freq[0,fi] 
                        phase   = - pos * k
                        replica = np.exp(1j*phase.transpose()) #/ np.linalg.norm(np.exp(1j*phase))
                        fk[[fi],U.index(ux),U.index(uy)]  = (np.conj(replica) * K[0,:,:].squeeze() * replica.T).squeeze()/Nb
                        arf[[fi],U.index(ux),U.index(uy)] = (np.conj(replica) * K0[0,:,:].squeeze() * replica.T ).squeeze()/Nb
                            
        if fktype=='mvdr':
            for fi in range(freq.size):
                #print(str(fi))
                for ux in U: 
                    for uy in U : 
                        ba     = baz[U.index(ux),U.index(uy)] 
                        if output_compo=='T':
                            dataRT  = - dataE * sin(ba) - dataN * cos(ba)
                        elif output_compo=='R':
                            dataRT  = - dataE * cos(ba) + dataN * sin(ba)
                        else:
                            raise ValueError("output_compo = 'T' or 'R'")
                        K,K0     = cross_spectral_mat(dataRT[[fi],:],replica0[[fi],:],Nb,fktype)                           
                        Kinvf = np.linalg.inv( K[0,:,:].squeeze() )
                        K0invf= np.linalg.inv( K0[0,:,:].squeeze() )                        
                        s       = np.matrix([ux,uy]).transpose()
                        k       = 2 * np.pi * s * freq[0,fi]
                        phase   = - pos * k                 
                        replica = np.exp(1j*phase.transpose()) #/ np.linalg.norm(np.exp(1j*phase))
                        fk[[fi],U.index(ux),U.index(uy)]  = (1/(np.conj(replica) * Kinvf * replica.T)).squeeze()/Nb
                        arf[[fi],U.index(ux),U.index(uy)] = (1/(np.conj(replica) * K0invf * replica.T)).squeeze()/Nb
    print("        Done in %.2f sec"%(time.time()-t0))
    return fk,arf

################################################################
################################################################
################################################################
################################################################
# BEAM SUBFCTS

def manage_station_list(in_,bb):
    sta   = io_beam.read_station_list(in_['station_file']) 
    list_of_keys = list(sta.keys())

    lat   = np.zeros([len(sta)])
    lon   = np.zeros([len(sta)])

    for kname in sta.keys():
        lat[list_of_keys.index(kname)] = sta[kname]['lat']
        lon[list_of_keys.index(kname)] = sta[kname]['lon']

    recenter =  False
    if bb['center']  is None:
        center = get_center(lat,lon)
        recenter = True
    else:
        center = bb['center'] 
        
    pos   = get_xy_from_center(lat,lon,center,ref_az=0)
    radius = np.sqrt(pos[:,0]**2+pos[:,1]**2)

    list_of_keys = list(sta.keys())

    if bb['max_radius_selection']:
        index_pos = radius < bb['max_radius_selection']
        pos  = pos[index_pos]
        lat  = lat[index_pos]
        lon  = lon[index_pos]
        tmp_sta = {}
        for idx in np.where(index_pos)[0]:
            tmp_sta[list_of_keys[idx]] = sta[list_of_keys[idx]]
        sta = tmp_sta 
        del tmp_sta
        list_of_keys = list(sta.keys())

    if recenter: #recompute array center and relative pos
        center = get_center(lat,lon)
        pos    = get_xy_from_center(lat,lon,center,ref_az=0)

    return {
        'sta':sta,
        'pos':pos,
        'lat':lat,
        'lon':lon,
        'list_of_keys':list_of_keys,
        'center':center}

def cross_spectral_mat(data,replica0,Nb,fktype):
    K  = (np.transpose(np.tile(np.conj(data), (1,Nb,1,1)),(0,2,3,1)) * np.transpose(np.tile(data, (1,Nb,1,1)),(0,2,1,3))/2).squeeze(axis=0)
    K0 = (np.transpose(np.tile(np.conj(replica0), (1,Nb,1,1)),(0,2,3,1)) * np.transpose(np.tile(replica0, (1,Nb,1,1)),(0,2,1,3))/2).squeeze(axis=0)

    for fi in range( replica0.shape[0] ):
        if fktype=='mvdr' :
            eyeK = K[[fi],:,:].squeeze()
            eyeK0 = K0[[fi],:,:].squeeze()
            K[[fi],:,:]=K[[fi],:,:] + 0.01*np.linalg.norm(K[[fi],:,:])* np.eye( eyeK.shape[0],eyeK.shape[0] )
            K0[[fi],:,:]=K0[[fi],:,:] + np.linalg.norm(K0[[fi],:,:])*0.01*np.eye( eyeK0.shape[0],eyeK0.shape[0] )
        K[[fi],:,:]= K[[fi],:,:] / np.linalg.norm(K[[fi],:,:])
        K0[[fi],:,:]=K0[[fi],:,:] / np.linalg.norm(K0[[fi],:,:])    
    return K,K0


def make_short_freq(f0,freq,fw=1,nf=0):
    if fw==0:
        ifreq = np.linspace(0,np.ceil(len(freq)/2)-1,int(np.ceil(len(freq)/2)),dtype=int)
        if nf:
            ifreq = ifreq[np.linspace(0,len(ifreq)-1,int(nf),dtype=int).tolist()]        
    elif fw==1:
        ifreq = np.array(np.abs(freq-f0).argmin())
    else:     
        ifreq = np.array(np.where((f0/fw<freq) & (freq<f0*fw))[0])
        if nf:
            ifreq = ifreq[np.linspace(0,len(ifreq)-1,int(nf),dtype=int).tolist()]
    return np.array(ifreq).tolist()


def make_short_freq_range(f1,f2,freq,nf=0): 
    ifreq = np.array(np.where((f1<freq) & (freq<f2))[0])
    if nf:
        ifreq = ifreq[np.linspace(0,len(ifreq)-1,int(nf),dtype=int).tolist()]
    return np.array(ifreq).tolist()


def get_xy_from_center(lat,lon,center,ref_az=0):
    pos = np.zeros([len(lat),2])
    for i in range(len(lat)):
        angle = gps2dist_azimuth(center[0],center[1],lat[i],lon[i])
        pos[i,0]=angle[0]*np.cos(np.deg2rad(90-angle[1] % 360))
        pos[i,1]=angle[0]*np.sin(np.deg2rad(90-angle[1] % 360))
    ref_az=np.pi*ref_az/180;
    pos[:,0] = pos[:,0]*np.cos(ref_az) + pos[:,0]*np.sin(ref_az) ;
    pos[:,1] = pos[:,1]*np.cos(ref_az) - pos[:,1]*np.sin(ref_az) ;
    return pos


def get_center(lat,lon):
    # on a sphere ...
    # need validation !!
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    x     = np.mean(np.cos(lat) * np.cos(lon));
    y     = np.mean(np.cos(lat) * np.sin(lon));
    z     = np.mean(np.sin(lat));
    rr    = np.sqrt(x**2 + y**2)
    mlat  = np.arctan2( z, rr) * 180 / np.pi
    mlon  = np.arctan2( y, x) * 180 / np.pi
    return np.array([mlat,mlon])