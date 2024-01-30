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
from cartopy.io.img_tiles import Stamen, OSM
import cartopy.feature as cfeature
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics import locations2degrees
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import m_beam
import io_beam
import dd
import lang


import cProfile
import ipdb


def make_subfig(beam_file,pl_arg={},num_colu=3,info='beam'):
    if pl_arg=={}:
        print('will use default plot arguments included within beam file')
        parse = False
        #otherwise need to parse ...   
    else:
        parse = True
    
    print(beam_file)
    if not os.path.isfile(beam_file):
        raise ValueError("not such file  " + beam_file)
    h5b   = h5.File(beam_file,'r')
    
    # load pl argument form input file and parse if necessary
    pl = io_beam.recursively_load_dic_from_h5(h5b,'/pl/')

    if parse:
        pl = lang.merge_options(pl,pl_arg)

    # number of subplots ?
    submap = 0
    if pl['map']:submap = 1
    nv_sub = len(pl['fn'])+submap
    print("number of subplots: %d"%nv_sub)
    
    num_line = (nv_sub-1)//num_colu+1
    sub = str(num_line) + str(num_colu)
    
    # define mesh from U
    U    = h5b['bb']['U'][()] * 10**3
    du   = U[1]-U[0]
    y, x = np.mgrid[min(U)-du/2:max(U)+du/2:U.size+1+1j,min(U)-du/2:max(U)+du/2:U.size+1+1j]    
    
    #load arf and fk
    arf  = h5b['arf'][()]
    fk   = h5b['fk'][()]
    freq = h5b['ff']['sfreq'][()]
    
    #loop over freq
    inc = 0
    for sub_freq in pl['fn']:
        # find ifreq for sub_freq and 
        inc+=1
        ifreq = m_beam.make_short_freq(float(sub_freq),freq,fw=pl['fw'],nf=0)
        Z     = np.mean(np.abs(fk[ifreq,:,:]),axis=0).transpose()
        Zarf  = np.mean(np.abs(arf[ifreq,:,:]),axis=0).transpose()
        Zarf  = Zarf - np.min(Zarf)       
        Zarf  = Zarf * gauss_2d(x=Zarf.shape[0],y=Zarf.shape[0],mu=0, sigma=pl['deconv']['sigma'])

        if pl['deconv']['apply']:
            filter_epsilon = None
            if pl['deconv']['filter_epsilon'] != None:
                filter_epsilon = pl['deconv']['filter_epsilon']
            Z = deconv.richardson_lucy(Z,Zarf,num_iter=pl['deconv']['iter'],clip=pl['deconv']['clip'],filter_epsilon=filter_epsilon)

        if pl['power_fk']:
            Z = Z**2
            Zarf = Zarf**2
        
        if pl['arf_only']:
            Z = Zarf

        if pl['norm']:
            Z = Z / np.max(Z)    
        
        if pl['db']:
            Z = 20 * np.log10(Z)

        y_ev = 0
        x_ev = 0
        try :
            baz = h5b['bb']['target_baz'][()]
            if baz != None:
                y_ev = np.array([0,np.sin(np.deg2rad(90-h5b['bb']['target_baz'][()] % 360))])*U.max()
                x_ev = np.array([0,np.cos(np.deg2rad(90-h5b['bb']['target_baz'][()] % 360))])*U.max()
        except:
            pass
        
        ylab=False
        if int(inc%num_colu)==1 : ylab=True
        xlab=False
        if inc > (num_line-1)*num_colu : xlab=True
        
        plot_beam(int(sub+str(inc)),Z,x,y,U,x_ev,y_ev,title="%.2f"%freq[ifreq[0]] + " - %.2f Hz"%freq[ifreq[-1]] ,pl=pl,xlab=xlab,ylab=ylab)
    
    mmap(int(sub+str(inc+1)),h5b['sta']['lon'][()],h5b['sta']['lat'][()],info=info)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fig = plt.gcf()
    #fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.resizable = True
    h5b.close()
    
    
def gauss_2d(x=10,y=10,mu=0, sigma=1):
    x, y = np.meshgrid(np.linspace(-1,1,x), np.linspace(-1,1,y))
    d = np.sqrt(x*x+y*y)
    return np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )


def mydeconv(Z,Zarf,eps=1):
    Zarf = scipy.fftpack.fft2(Zarf)
    #Zarf =  np.abs(Zarf / np.linalg.norm(Zarf) * 100)
    Z = scipy.fftpack.fft2(Z) / (Zarf+ eps)
    return scipy.fftpack.ifft2(Z).real


def plot_beam(sub,Z,x,y,U,x_ev=0,y_ev=0,title='',pl={'clmap': 'viridis','clim':None,'target':{'dU':None,'color':'w'}},xlab=False,ylab=False):
    ax     = plt.subplot(sub)
    
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    
    if pl['clim'] is None:
        cc = [Z.min(),Z.max()]
    else:
        cc = pl['clim']
    if pl['target']['dU'] is None:
        pl['target']['dU'] = np.max(U)/3
    im     = ax.imshow(Z,vmin=cc[0], vmax=cc[1],origin='lower',extent=extent,cmap=pl['clmap'])
    plt.title(title)
    make_target(U,ax,dr=pl['target']['dU'],color=pl['target']['color'])
    if x_ev!=0 or y_ev!=0:
        plt.plot([0,x_ev],[0, y_ev],color='r')
    
    if pl['xy_zoom'] != None:
        plt.xlim(pl['xy_zoom'][0], pl['xy_zoom'][1])
        plt.ylim(pl['xy_zoom'][2], pl['xy_zoom'][3])
    
    if xlab:plt.xlabel("slowness (s/km)", size='small')
    if ylab:plt.ylabel("slowness (s/km)", size='small')
    
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="5%", pad="2%")
    acx = plt.colorbar(im, cax=cax)
    acx.ax.tick_params(labelsize=5)

    
def mmap(sub,lon=[2.39,5.72],lat=[47.08,45.18],filename='stations.png',info='station map',xylabel=False,res=None):
    imagery = OSM()
    ax = plt.subplot(sub,projection=imagery.crs)
    #stamen_terrain = Stamen('terrain')
    #ax     = plt.subplot(sub,projection=stamen_terrain.crs)
    minlat = -80
    maxlat = 80
    dl     = locations2degrees(min(lat),min(lon),max(lat),max(lon))*0.05
    minlon = -180+dl/2
    maxlon = 180-dl/2
    r      = int(locations2degrees(min(lat),min(lon),max(lat),max(lon))*111)
    if r < 10000:
        if min(lat) - dl > minlat: minlat = min(lat) - dl
        if max(lat) + dl < maxlat: maxlat = max(lat) + dl
        if min(lon) - dl > minlon: minlon = min(lon) - dl
        if max(lon) + dl < maxlon: maxlon = max(lon) + dl
    ax.set_extent([minlon, maxlon, minlat, maxlat])
    r = int(locations2degrees(minlat,minlon,maxlat,maxlon)*111)
    try :
        if res==None:
            if r <= 50:
                res = 15 
                ax.coastlines('10m')
                #scale_bar(ax,r/10-r/10%10)
            if r > 50 and r <= 1000 :
                res = 10 
                ax.coastlines('10m')
                #scale_bar(ax,r/10-r/10%10)
            if r > 1000 and r <= 5000:
                res = 8
                ax.coastlines('50m')
                #scale_bar(ax,r/10-r/10%10)  
            if r > 5000 and r <= 10000: 
                res = 6
                ax.coastlines('110m')
            if r > 10000: 
                res = 3
                ax.coastlines('110m')
        ax.add_image(imagery,res)
        gl = ax.gridlines(draw_labels=False,linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels   = xylabel
        gl.left_labels  = xylabel
    except:
        pass
    plt.scatter(lon, lat, c='red', s=20,transform=ccrs.PlateCarree(),cmap='plasma_r',alpha=0.5)
    plt.title(info)


def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3):
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    x0, x1, y0, y1 = ax.get_extent(tmc)
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 
    bar_xs = [sbx - length * 500, sbx + length * 500]
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')


def make_target(U,ax,dr=0.1,color='w'):
    ax.plot(U,U,color=color)
    ax.plot(U,-U,color=color)
    ax.plot(U,np.zeros(U.shape),color=color)
    ax.plot(np.zeros(U.shape),U,color=color)
    for ic in range(int(max(U)/dr)):
        circ=plt.Circle((0,0), radius=dr*(ic+1), color=color, fill=False)
        ax.add_patch(circ)
        
