import m_pycorr.mods.dd as dd 
import m_pycorr.live.mods.h5 as h5 
from m_pycorr.live.c1_trace import c1_trace as c1_trace  

import os , glob , h5py, pickle
import numpy as np 
import sys
import random
import glob
import copy
import scipy.fftpack
import scipy.signal as signal
import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from obspy.geodetics import gps2dist_azimuth as gps2dist
from obspy.geodetics import kilometer2degrees
from obspy.geodetics import degrees2kilometers
from obspy.geodetics import locations2degrees
from obspy.core import UTCDateTime
from matplotlib.transforms import offset_copy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
try :
    import ipdb 
except:
    pass 

#-----------------------------------------------------
# Typical usage : 
# import m_live
# h1 = m_live.c1_md('C1_04_xcorr_test__xcorr_norm')
# h1.plot_station_map()
# [...]
#
# ----------------------------------------------------
# READING methods : 
#
# read_c1_for_path_between_date : 
#   use read_c1 to read all c1 between id1-id2 or id2-id1 time reversed btw date1 and date2
#   and sum them all w/o normalization. 
#   return c1_trace class instance, is_data=[True/False]
# read_c1_for_path : 
#   use read_c1 to read the c1 between id1-id2 or id2-id1 time_reversed.
#   retrun c1_trace class instance , is_data=[True/False]
#
# read_c1 : 
#   most basic funtions !
#   read the c1 for a path indice, cmp, date. 
#   return a c1_trace class instance with the c1 trace ,c1 metadata, title 
#
#....
#---------------------------------------------------------------------------------
#
# UTILITY methods :
# ....
# get_station_list : return a dict containing the metadata of all stations
#----------------------------------------------------------------------------------------- 
#
# CONSTRUCTOR : 
#
# -the 1st time: scann all C3_*/xcorr_set*h5 files and concatenate their metadata in C3_*/db.pkl file 
# -next time : re-read the db.pkl file 
#-----------------------------------------------------------------------------------------------------


class c1_md : 
    '''
    When run the 1st time, it scans all correlations and create a db.pkl file  containing their metadata. 
    '''
    def get_selection(self,idvs=[],idvr=[],icmp=0,tmin=None,tmax=None,
        dmin=None,dmax=None, filter_p=False,p1=10.,p2=100.,filter_f=False,f1=10.,f2=100.,
        ctype='NP',norm_tr=False,bin_distance=False,bin_width=1.,save=True,frmt='mat',
        file_name='./test.mat',dist_unit='km',time_unit='s',iddate=-1,slant_stack={'type':'off','u':1/3.,
        'sum':{'type':'linear','timegate':20,'power':2}}) :
        ''' save or get a selection of correlations and the corresponding metadata as pkl or mat file
            default args :
            idvs=[], idvr=[]    : id virtual source(s)/receiver(s), as list of station names
                                  for instance 
                                  idvs = ['NET1.STA1.LOC','NET1.STA2.LOC'...]
                                  idvr = ['NET1.STA1.LOC','NET1.STA2.LOC'...]
                                  both idvs and idvr accept wildcard * as last character : ['NET1.*', 'NET2.ST*',...]
                                  default values are empty list [] for both. In that case, idsv will be a random station
                                  idvr the full list of available station.
        
            icmp = 0            : scalar, index of the component as in self.in_['cc_cmp'] or self.in_rot['cc_cmp'] if self.rtz=True
            filter_p = Flase    : boolean, switch on/off a band-pass filter in period
            p1 = 10.            : scalar, period min if filter_p=True
            p2 = 100.           : scalar, period max if filter_p=True
            filter_f = False    : boolean, switch on/off a band-pass filter in frequency
            f1 = 10.            : scalar, freq min if filter_f=True
            f2 = 100.           : scalar, freq max if filter_f=True
            ctype = 'NP'        : srting, 'NP' for full correlation, 'N' or 'P' for negative 
                                           or positive part respectively, 'S' for symmetric part
            norm_tr = False     : boolean, switch on/off normalize each trace by its max(abs())        
            bin_distance = True : boolean, switch on/off bin distance axis
            bin_width = 1.      : scalar, bin size in dist_unit, if bin_distance
            tmin/max = None     : scalar, min/max lag-time of the output, default corresponds to full scale
            dmin/max = None     : scalar, min/max distance of the output, default corresponds to full scale
            save = True         : boolean, switch on/off save as file instead of returning
            frmt = 'mat'        : string, output format, mat or pkl 
            file_name = './test.mat' : string, path/filename.ext full name for the figure. Beware the format 
            dist_unit = 'km'    : string, unit for distance and bin axes
             : 'km', 'm' or 'deg'
            time_unit = 's'     : string, unit for time axis : 's', 'min' or 'h'
            iddate    = -1      : id of the date from 0 to len(h1.md_c['date*']), -1 means ref !
            slant_stack ={'type':'off','u':1/6.,'sum'={'type':'linear','timegate':None,'power':None}}:
                        type : 'off', do not apply,  'single' returns a single slant_stack trace and 'vespa' for full vespagram
                        u : for slowness, a single float for 'single' mode and a np.array for 'vespa'
                        sum : 'type' can e 'linear' or 'pws'. For 'pws', 'timegate' and 'power' are float values
        '''
        opt = {}
        opt['file_name']    = file_name
        opt['icmp']         = icmp
        opt['filter_p']     = filter_p
        opt['p']            = [p1,p2]
        opt['filter_f']     = filter_f
        opt['f']            = [f1,f2]
        opt['ctype']        = ctype
        opt['save']         = save
        opt['format']       = frmt
        opt['norm_tr']      = norm_tr
        opt['tlim']         = [tmin,tmax]
        opt['dlim']         = [dmin,dmax]
        opt['bin_distance'] = bin_distance
        opt['bin_width']    = bin_width
        opt['dist_unit']    = dist_unit
        opt['time_unit']    = time_unit
        opt['iddate']       = iddate
        opt['slant_stack']  = slant_stack
        vs = self._idv(idv=idvs)
        vr = self._idv(idv=idvr,inv='r')
        if vs == [] or vr == []:
            dd.dispc('check idvs and/or idvr ...','r','d')
            return
        data,time,dist,opt  = self._load_idv(vs,vr,opt)
        if opt['tlim'][0] is None: opt['tlim'][0]=time[0]
        if opt['tlim'][1] is None: opt['tlim'][1]=time[-1]
        if opt['dlim'][0] is None: opt['dlim'][0]=min(dist)
        if opt['dlim'][1] is None: opt['dlim'][1]=max(dist)
        time_select = (time >= opt['tlim'][0]) & (time <= opt['tlim'][1])
        data        = data[time_select,:]
        time        = time[time_select]
        if opt['save'] : self._save_idv(data,time,dist,vs,vr,opt)
        else: return self._get_idv(data,time,dist,vs,vr,opt) 


    def plot_selection(self,idvs=[],idvr=[],icmp=0,lag=None,tmin=None,tmax=None,
        dmin=None,dmax=None,filter_p=False,p1=10.,p2=100.,filter_f=False,f1=10.,f2=100.,
        ctype='NP',norm=1,norm_tr=False,bin_distance=True,bin_width=1.,cmap='gray',
        save_plot=False,file_name='./test.png',dist_unit='km',time_unit='s',iddate=-1):
        ''' plot a selection of correlations.
            default args :
            idvs=[], idvr=[]    : id virtual source(s)/receiver(s), as list of station names
                                  for instance 
                                  idvs = ['NET1.STA1.LOC','NET1.STA2.LOC'...]
                                  idvr = ['NET1.STA1.LOC','NET1.STA2.LOC'...]
                                  both idvs and idvr accept wildcard * as last character : ['NET1.*', 'NET2.ST*',...]
                                  default values are empty list [] for both. In that case, idsv will be a random station
                                  idvr the full list of available station.
        
            icmp = 0            : scalar, index of the component as in in_['cc_cmp']
            lag  = None         : scalar, lag-time to plot (unit is defined by time_unit), default None=>full range
            tmin/max = None     : scalar, min/max lag-time of the output, default corresponds to full scale
            dmin/max = None     : scalar, min/max distance of the output, default corresponds to full scale
            filter_p = Flase    : boolean, switch on/off a band-pass filter in period
            p1 = 10.            : scalar, period min if filter_p=True
            p2 = 100.           : scalar, period max if filter_p=True
            filter_f = False    : boolean, switch on/off a band-pass filter in frequency
            f1 = 10.            : scalar, freq min if filter_f=True
            f2 = 100.           : scalar, freq max if filter_f=True
            ctype = 'NP'        : srting, 'NP' for full correlation, 'N' or 'P' for negative 
                                           or positive part respectively, 'S' for symmetric part
            norm  = 1           : scalar, a normalization factor. 
                                           for wiggle plot : tr=tr*norm
                                           for pcolor : color limits = +/- std(data)*norm (opposite effect...)
            norm_tr = False     : boolean, switch on/off normalize each trace by its max(abs())        
            bin_distance = True : boolean, switch on/off bin distance axis
            bin_width = 1.      : scalar, bin size in dist_unit, if bin_distance
            cmap = 'gray'       : string, colormap for pcolor
            save_plot = False   : boolean, switch on/off save a figure as a file_name instaed of showing ...
            file_name = './test.png' : string, path/filename.ext full name for the figure. It includes format
            dist_unit = 'km'    : string, unit for distance and bin axes : 'km', 'm' or 'deg'
            time_unit = 's'     : string, unit for time axis : 's', 'min' or 'h'
            iddate    = -1      : id of the date from 0 to len(h1.md_c['date*']), -1 means ref !
            TO DO : finish tmin/max, dmin/max selection
            do azimuthal selection
            do better default values (lag, p1, p2...) depending on the input
        '''
        opt = {}
        opt['icmp']         = icmp
        opt['filter_p']       = filter_p
        opt['p']            = [p1,p2]
        opt['filter_f']       = filter_f
        opt['f']            = [f1,f2]
        opt['ctype']        = ctype
        opt['lag']          = lag       
        opt['tlim']         = [tmin,tmax]
        opt['dlim']         = [dmin,dmax]
        opt['norm']         = norm
        opt['norm_tr']      = norm_tr
        opt['cmap']         = cmap
        opt['save_plot']    = save_plot
        opt['file_name']    = file_name
        opt['bin_distance'] = bin_distance
        opt['bin_width']    = bin_width
        opt['dist_unit']    = dist_unit
        opt['time_unit']    = time_unit
        opt['iddate']       = iddate
        vs  = self._idv(idv =idvs)
        vr  = self._idv(idv =idvr,inv='r')
        if vs == [] or vr == []:
            dd.dispc('check idvs and/or idvr ...','r','d')
            return
        data,time,dist,opt  = self._load_idv(vs,vr,opt)
        if opt['tlim'][0] is None: opt['tlim'][0]=time[0]
        if opt['tlim'][1] is None: opt['tlim'][1]=time[-1]
        if opt['dlim'][0] is None: opt['dlim'][0]=min(dist)
        if opt['dlim'][1] is None: opt['dlim'][1]=max(dist)
        time_select = (time >= opt['tlim'][0]) & (time <= opt['tlim'][1])
        data        = data[time_select,:]
        time        = time[time_select]
        self._plot_idv(data,time,dist,vs,vr,opt)
        return

    def plot_path(self,id1=None,id2=None,V_cmp=range(9),lag=3600,filter_p=False,p1=10.,p2=100.,
        filter_f=False,f1=10.,f2=100.,ctype='NP',norm=1,norm_tr=False,save_plot=False,
        file_name='./test.png',time_unit='s') :
        opt = {}
        opt['id1']       = id1
        opt['id2']       = id2
        opt['V_cmp']     = V_cmp
        opt['lag']       = lag
        opt['p']         = [p1,p2]
        opt['filter_f']  = filter_f
        opt['f']         = [f1,f2]
        opt['ctype']     = ctype
        opt['norm']      = norm
        opt['norm_tr']   = norm_tr
        opt['save_plot'] = save_plot
        opt['file_name'] = file_name
        opt['time_unit'] = time_unit        
        data,time,cmpaxis,opt = self._load_path(id1,id2,opt)
        id1=opt['id1']
        id2=opt['id2']
        if not data.max(): 
            print('No such correlation: ' + id1 + ' ' + id2)
            return
        else: self._plot_idpath(data,time,cmpaxis,opt)
        return

    def plot_station_map(self,map_show=1, ax = [],idvs=[],idvr=['all'],color_vs='r',color_vr='b',size_vs=4,size_vr=6) :
        ''' Plot station map in axis handle axs
            By default it only plots a set of receivers which is set to 'all', as blue dotes.
            idvs and idvr allow to select different virtual sources/receivers (vs/vr) respectively
            warning : figure displays only if map_show==1
            
            default args :
            
            map_show  = 1            : 0/1 show map ...
            idvs =[], idvr=['all']   : id virtual source(s)/receiver(s), as list of station names
                                      for instance 
                                      idvs = ['NET1.STA1.LOC','NET1.STA2.LOC'...]
                                      idvr = ['NET1.STA1.LOC','NET1.STA2.LOC'...]
            color_vs='r',color_vr='b': marker face color for vs and vr 
            size_vs=4, size_vr=6     : marker size
        '''
        if ax == []:
             ax = plt.axes(projection=Stamen('terrain-background').crs)
        lat   = self.get_station_list()['lat']
        lon   = self.get_station_list()['lon']
        idsta = list(self.get_station_list()['id'])    
        minlat = -90 
        maxlat = 89
        minlon = -180
        maxlon = 180
        dl_lat=abs(max(lat) - min(lat))*0.1
        dl_lon=abs(max(lon) - min(lon))*0.1

        dl_tmp = max([dl_lon,dl_lat])
        dl_lat = dl_tmp
        dl_lon = dl_tmp

        dl     = 0.01
        if min(lat) - dl_lat > minlat: minlat = min(lat) - dl_lat
        if max(lat) + dl_lat < maxlat: maxlat = max(lat) + dl_lat
        if min(lon) - dl_lon > minlon: minlon = min(lon) - dl_lon
        if max(lon) + dl_lon < maxlon: maxlon = max(lon) + dl_lon
        ax.set_extent([minlon, maxlon, minlat, maxlat])
        r = int(locations2degrees(minlat,minlon,maxlat,maxlon)*111)
        if r <= 1000 :
            res = 8 
            ax.coastlines('10m')
        if r > 1000 and r <= 5000:
            res = 6
            ax.coastlines('50m')    
        if r > 5000 and r <= 10000: 
            res = 4
            ax.coastlines('110m')
        if r > 10000: 
            res = 2
            ax.coastlines('110m')
        ax.add_image(Stamen('terrain-background'),res)
        ax.gridlines(draw_labels=False)

        if idvr[0]=='all':
            for ista in range(0,len(lat)):
                ax.plot(lon[ista],lat[ista], "o", color=color_vr, 
                    markeredgecolor = 'k',markersize=size_vr,transform=ccrs.PlateCarree(),alpha=0.5)
                    #markeredgecolor = 'k',markersize=size_vr,transform=ccrs.Geodetic(),alpha=0.5)
        else:
            for iidvr in idvr:
                ax.plot(lon[idsta.index(iidvr)],lat[idsta.index(iidvr)], "o", color=color_vr, 
                    #markeredgecolor = 'k',markersize=7,transform=ccrs.Geodetic(),alpha=0.5)
                    markeredgecolor = 'k',markersize=7,transform=ccrs.PlateCarree(),alpha=0.5)
        if idvs != []:
            for iidvs in idvs:
                ax.plot(lon[idsta.index(iidvs)],lat[idsta.index(iidvs)], "o", color=color_vs, 
                    #markeredgecolor = 'k',markersize=5,transform=ccrs.Geodetic(),alpha=0.5)
                    markeredgecolor = 'k',markersize=5,transform=ccrs.PlateCarree(),alpha=0.5)

        if map_show==1:
            plt.show()
        return 


    def plot_cc_path(self) :
        print("to be done") 
        return

    def get_cc_path(self):
        print("to be done") 
        return

    def get_station_list(self) :
        ''' return a dict containing the metadata of all stations
        '''
        sta={}
        [sta['id'], I]=np.unique(self.id.flatten(),return_index=True)
        sta['lat']    =self.lat.flatten()[I] 
        sta['lon']    =self.lon.flatten()[I] 
        sta['elev']   =self.elev.flatten()[I]
        return sta 

    def read_c1_for_path_between_date(self,id1,id2,date1,date2,I_cmp=0) :
        ''' read all c1 between id1 and id2 at all date between date1 and date2, and sum them.
          if the c1 id1-id2 do not exist, read id2-id1 and time reverse it 
          id1, id2    : id of both station 
          date1,date2 : UTCDatetime date
          I_cmp       : indice of the components [ZZ,ZN,...] to be read 

        '''
        nwf   = len(self.md_c['t'])
        c1_tr = np.zeros((nwf))
        #0. get indice of all correlations starting after date1 and ending before date2 
        I = np.intersect1d(np.where(self.md_c['date1']>=date1), np.where(self.md_c['date2']<=date2))
        #1. read them all and sum them :
        norm  = 0
        for I_date in I : 
            tr, is_data = self.read_c1_for_path(id1,id2,I_cmp=I_cmp,I_date=I_date)
            if is_data :
                norm += 1.
                c1_tr += tr.tr
        # read the reference to get the metadata and substuting the trace by c1_tr: 
        c1, is_data = self.read_c1_for_path(id1,id2,I_cmp=I_cmp,I_date=-1)
        if is_data :
            c1.tr = c1_tr/norm 
            return c1, True 
        else : 
            return [], False 

    def read_c1_for_path_for_dates(self,id1,id2,list_I_date=[-1],I_cmp=0) :
        ''' read all c1 for the list of dates correspondong to list_I_date, and sum them.
          if the c1 id1-id2 do not exist, read id2-id1 and time reverse it 
          id1, id2    : id of both station 
          list_I_date : list of date indices. (-1 = reference)
          I_cmp       : indice of the components [ZZ,ZN,...] to be read 

        '''
        nwf   = len(self.md_c['t'])
        c1_tr = np.zeros((nwf))
        norm  = 0
        #1. read them all and sum them :
        for I_date in list_I_date : 
            tr, is_data = self.read_c1_for_path(id1,id2,I_cmp=I_cmp,I_date=I_date)
            if is_data :
                norm += 1.
                c1_tr += tr.tr
        # read the reference to get the metadata and substuting the trace by c1_tr: 
        c1, is_data = self.read_c1_for_path(id1,id2,I_cmp=I_cmp,I_date=-1)
        if is_data :
            c1.tr = c1_tr/norm 
            return c1, True 
        else : 
            return [], False 

    def read_c1_for_path(self,id1,id2,I_cmp=0,I_date=-1) : 
        ''' read c1 between id1 and id2. If the c1 id1-id2 does not exist it reads the c1 id2-id1 and time reverse it.
            id1 = CH.TORNY.00 
            id2 = CH.SLE.00 
            return an empty dict if c1 was not found 
        '''
        I1=np.intersect1d(np.where(self.id[:,0]==id1), np.where(self.id[:,1]==id2))
        I2=np.intersect1d(np.where(self.id[:,0]==id2), np.where(self.id[:,1]==id1))
        if len(I1) > 0 :
            I=I1[0] 
            reverse = False  
            c1_found= True 
        elif len(I2) > 0 :
            I=I2[0]
            reverse = True
            c1_found = True 
        else : 
            c1_found = False 

        if c1_found :
            tr=self.read_c1(I,I_cmp=I_cmp,I_date=I_date,reverse=reverse) 
            return tr,True 
        else : 
            return [],False

    def read_c1(self,I_path,I_cmp=0,I_date=-1,reverse=False)  : 
        ''' read a single correlate 
            I_path : indice of the path 
            I_cmp  : indice of the component 
            I_date : indice of the date. -1 = reference 
            reverse: time reverse the correlations and metadata 
                      (useful for c3 computation)
        '''
        if self.rtz: 
            #ipdb.set_trace()
            ccmp = self.in_rot['cc_cmp'][I_cmp].decode('utf8')
        else:
            ccmp = self.in_['cc_cmp'][I_cmp].decode('utf8')

        I_file = self._get_file_number(I_path)
        ff = h5py.File(self.c1_file[I_file],'r')
        dset_name = '/'+self.id[I_path][0]+'/'+self.id[I_path][1]
        dset_name = dset_name + '/' + ccmp
        tr={}
        if I_date == -1 : # reading the reference : 
            tr['tr']=ff['/ref'+dset_name][:]
            tr['date1'] = self.md_c['date1'][0]
            tr['date2'] = self.md_c['date2'][-1]

        else  :
            tr['tr']=ff['/cc'+dset_name][I_date,:]
            tr['date1'] = self.md_c['date1'][I_date]
            tr['date2'] = self.md_c['date2'][I_date]

        ff.close()
        if reverse : 
            I=[1,0] 
            tr['tr'] = np.flipud(tr['tr'])
            tr['baz']    = self.az[I_path] 
            tr['az']     = self.baz[I_path]
        else  :
            I=[0,1]
            tr['az']    = self.az[I_path] 
            tr['baz']   = self.baz[I_path]

        tr['title'] = self.id[I_path][I[0]]+'-'+self.id[I_path][I[1]]+' '+ccmp
        tr['cmp']   = ccmp
        tr['lon']   = self.lon[I_path][I] 
        tr['lat']   = self.lat[I_path][I]
        tr['id']    = self.id[I_path][I] 
        tr['dist']  = self.dist[I_path]

        tr['elev']  = self.elev[I_path][I] 
        tr['file']  = self.c1_file[I_file] 
        tr['time']  = self.md_c['t']
        tr['tau']   = self.md_c['tau']
        tr['tr'][np.isinf(tr['tr'])]=0.;
        return c1_trace(tr)


#########################################################################################
################################################################################ PRIVATE
#########################################################################################

    def _load_path(self,id1,id2,opt={}) :
        if not bool(opt): 
            print('no options')
            return
        if id1==None:opt['id1']=self.id[random.randint(0,len(self.id)-1),0]
        if id2==None:opt['id2']=self.id[random.randint(0,len(self.id)-1),1]
        return self._load_id_data_path(opt['id1'],opt['id2'],opt)

    def _load_idv(self,idvs,idvr,opt={}) :
        if not bool(opt): 
            print('no options')
            return
        idvs  = list(idvs)
        idvr  = list(idvr)
        IVS   = self._find_id(idvs,self.id)
        IV    = np.array(())
        for index, couple in enumerate(self.id[IVS.astype('int')]):
            if (couple[0] in idvs and couple[1] in idvr) or (couple[0] in idvr and couple[1] in idvs):
                IV = np.append(IV,IVS[index])
        IV = IV.astype('int')
        vdist = self.dist[IV]

        if opt['dist_unit']=='deg': vdist = kilometer2degrees(vdist)
        if opt['dist_unit']=='m':   vdist = vdist*1000
        
        if opt['dlim'][0] is not None:
            IV    = IV[np.where(vdist>=opt['dlim'][0])[0]]
            vdist = vdist[np.where(vdist>=opt['dlim'][0])[0]]
        if opt['dlim'][1] is not None:
            IV = IV[np.where(vdist<=opt['dlim'][1])[0]]
            vdist = vdist[np.where(vdist<=opt['dlim'][1])[0]]
        
        if not IV.any():
            print("no correlation(s)")
            return
        opt['az']  = self.az[IV]
        opt['baz'] = self.baz[IV]    
        opt['lat'] = self.lat[IV]
        opt['lon'] = self.lon[IV]
        opt['id']    = self.id[IV]
        opt['elev']  = self.elev[IV]
        opt['depth'] = self.depth[IV]
        return self._load_id_data_selection(IV,idvs,idvr,opt)

    def _load_id_data_path(self,id1,id2,opt):
        # define proper time vector depending on ctype
        if opt['ctype']=='NP':
            time = self.md_c['t']
        else:
            time = self.md_c['t'][round(len(self.md_c['t'])/2):]
        if opt['time_unit']=='min':time = time/60.
        if opt['time_unit']=='h':time = time/3600.
        # load data
        data = np.zeros((len(time),len(opt['V_cmp'])))
        cmpaxis = []
        for icmp in opt['V_cmp']:
            if type(opt['iddate'])==list:
                d   = self.read_c1_for_path_for_dates(id1,id2,I_cmp=icmp,list_I_date=opt['iddate'])[0]
            else:
                d   = self.read_c1_for_path(id1,id2,I_cmp=icmp,I_date = opt['iddate'])[0]
            if d!=[]:
                if opt['filter_p'] and len(opt['p'])==2:
                    d.filter(opt['p'][0],opt['p'][1])
                if opt['filter_f'] and len(opt['f'])==2:
                    d.filter(1./opt['f'][1],1./opt['f'][0])
                if opt['ctype']=='NP':trace = d.tr
                if opt['ctype']=='N' :trace = d.split_ca()['N']['trace']
                if opt['ctype']=='P' :trace = d.split_ca()['P']['trace']
                if opt['ctype']=='S' :trace = d.split_ca()['S']['trace']
                data[:,opt['V_cmp'].index(icmp)] = trace
                cmpaxis.append(d.cmp)
            else:
                cmpaxis.append('')
        return data,time,cmpaxis,opt

    def _load_id_data_selection(self,I,idvs,idvr,opt):
        # define proper time vector depending on ctype
        if opt['ctype']=='NP':
            time = self.md_c['t']
        else: 
            time = self.md_c['t'][round(len(self.md_c['t'])/2):]
        if opt['time_unit']=='min':time = time/60.
        if opt['time_unit']=='h':time = time/3600.
        # load data
        data = np.zeros((len(time),len(I)))
        dist = np.zeros(len(I))
        inc  = 0
        num_cc = len(I)
        dd.dispc('extracting ' + str(num_cc) + ' cc','b','b')
        for icorr in I.astype('int'):
            inc+=1.
            perc = inc * 100. / num_cc
            if perc % 10 == 0:
                dd.dispc(str(perc) + '%','b','b')
            if self.id[icorr,0] not in idvs and self.id[icorr,0] in idvr:
                id1 = self.id[icorr,1]
                id2 = self.id[icorr,0]
            else:
                id1 = self.id[icorr,0]
                id2 = self.id[icorr,1]
            #ipdb.set_trace()
            if type(opt['iddate'])==list:
                d   = self.read_c1_for_path_for_dates(id1,id2,I_cmp=opt['icmp'],list_I_date=opt['iddate'])[0]
            else:
                d   = self.read_c1_for_path(id1,id2,I_cmp=opt['icmp'],I_date = opt['iddate'])[0]
            if d!=[]:
                if opt['filter_p'] and len(opt['p'])==2:
                    d.filter(opt['p'][0],opt['p'][1])
                if opt['filter_f'] and len(opt['f'])==2:
                    d.filter(1./opt['f'][1],1./opt['f'][0])
                if opt['ctype']=='NP':trace = d.tr
                if opt['ctype']=='N' :trace = d.split_ca()['N']['trace']
                if opt['ctype']=='P' :trace = d.split_ca()['P']['trace']
                if opt['ctype']=='S' :trace = d.split_ca()['S']['trace']
                data[:,list(I).index(icorr)] = trace
                dist[list(I).index(icorr)]   = d.dist
        if opt['dist_unit']=='deg':dist=kilometer2degrees(dist)
        if opt['dist_unit']=='m':dist = dist*1000.
        return data,time,dist,opt

    def _plot_idpath(self,data,time,cmpaxis,opt):     
        #fig    = plt.figure(figsize=(6, 7))
        ax_map = plt.subplot(2, 1, 1, projection=Stamen('terrain-background').crs)
        ax0    = plt.subplot(2, 1, 2)
        for icmp in opt['V_cmp']:
            if d.tr.max():
                MM = max(2*abs(d.tr))
            else:MM=1
            ax0.plot(self.md_c['t'],d.tr /MM + opt['V_cmp'].index(icmp),'k')
            ok = True
            print(d.tr)
        ax0.set_yticks(opt['V_cmp'])
        ax0.set_yticklabels(cch, rotation='horizontal', fontsize=8)
        ax0.grid(True)
        ax0.set_ylim(0.5,len(cch)+0.5)
        ax0.set_xlim(-lag,+lag)
        if opt['filter_p']:
            ax0.set_title(d.title + ' -- filter: ' + str(p1) + '- ' + str(p2) + ' s')
        elif opt['filter_f']:
            ax0.set_title(d.title + ' -- filter: ' + str(f1) + '- ' + str(f2) + ' hz')
        else:
            ax0.set_title(d.title + ' -- no filter')
        self.plot_station_map(map_show=0,ax=ax_map,idvs=[id1],idvr=[id2])
        plt.show()

    def _plot_idv(self,data,time,dist,idvs,idvr,opt) :
        fig      = plt.figure(figsize=(6, 7))
        ax_map   = fig.add_subplot(2, 1, 1, projection=Stamen('terrain-background').crs)
        ax_sect  = fig.add_subplot(2, 1, 2)
        self.plot_station_map(map_show=0,ax=ax_map,idvs=idvs,idvr=idvr)
        ax = self._plot_section(data,time,dist,ax=ax_sect,opt=opt) 
        if len(idvs)==1 : title = idvs[0]
        else : title = 'average'
        if self.rtz: ccmp = self.in_rot['cc_cmp'][opt['icmp']].decode('utf8')
        else:ccmp = self.in_['cc_cmp'][opt['icmp']].decode('utf8')
        if opt['filter_p']:
            ax.set_title(title + ' -- ' + ccmp + 
                ' -- filter: ' + str(opt['p'][0]) + '- ' + str(opt['p'][1]) + ' s')
        elif opt['filter_f']:
            ax.set_title(title + ' -- ' + ccmp + 
                ' -- filter: ' + str(opt['f'][0]) + '- ' + str(opt['f'][1]) + ' hz')
        else:
            ax.set_title(title + ' -- ' + ccmp + ' -- no filter')
        ax.set_xlabel('Time (' + opt['time_unit'] + ')')
        ax.set_ylabel('Distance (' + opt['dist_unit'] + ')' )
        if opt['save_plot']:
            mkdir(os.path.dirname(opt['file_name']))
            plt.savefig(opt['file_name'],dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None, 
            transparent=False, bbox_inches='tight', pad_inches=0.1,
            frameon=None)
        else:plt.show()
        return

    def _plot_section(self,data,time,dist,ax=[],opt={}) :
        if not bool(opt): 
            print('no options')
            return
        if ax ==[]:
            ax = plt.subplot(111)
        ncorr = data.shape[1]
        if opt['bin_distance']:
            bdata,bin_axis,bn = self._bin_data(data,time,dist,opt)
            if ncorr<50 :
                for idx,tr in enumerate(bdata.transpose()):
                    if bn[idx]:
                        ax.plot(time,tr*opt['norm']+bin_axis[idx],color='k')
            else:
                dt = self.md_c['tau']
                db = bin_axis[1] - bin_axis[0]
                slt  = slice(min(time)- dt/2, max(time) + dt/2, dt)
                slb  = slice(min(bin_axis) - db/2, max(bin_axis) + db/2,db)
                y, x = np.mgrid[slb,slt]
                ax.pcolor(x,y,bdata.transpose(),cmap=opt['cmap'],vmin=-np.nanstd(bdata)*opt['norm'],vmax=np.nanstd(bdata)*opt['norm'])
                ax.autoscale()
        else : 
            if ncorr > 50 : print('it is recommended to use bin_distance=True') 
            for idx,tr in enumerate(data.transpose()):
                if opt['norm_tr'] and abs(tr).max(): tr=tr/abs(tr).max()
                ax.plot(time,tr*opt['norm']+dist[idx],color='k')
        if opt['lag'] == None:opt['lag']=self.md_c['tau']*(len(time)-1)
        if opt['ctype']=='NP':ax.set_xlim(-opt['lag'],+opt['lag'])
        else:ax.set_xlim(0,+opt['lag'])
        if len(dist)==1:ax.set_ylim(dist[0]-1,dist[0]+1)
        else:ax.set_ylim(min(dist),max(dist))
        ax.grid(True)
        return ax

    def _save_idv(self,data,time,dist,idvs,idvr,opt) :
        dico = self._get_idv(data,time,dist,idvs,idvr,opt)
        mkdir(os.path.dirname(opt['file_name']))
        #ipdb.set_trace()
        #for kname in opt:
        #    dico[kname]= opt[kname]
        dico['opt']= opt
        if opt['format']=='mat':
            io.savemat(opt['file_name'],dico)
        elif opt['format']=='pkl':
            save_as_pkl(dico,opt['file_name'])
        else:
            print('format should be "mat" or "pkl"')      
        return

    def _get_idv(self,data,time,dist,idvs,idvr,opt) :
        dico = {}
        if opt['bin_distance']:
            dico['data'],dico['dist'],dico['bn'] = self._bin_data(data,time,dist,opt)
            if opt['slant_stack']['type']=='single':
                dico['slantstack'] = self._slant_stack(dico['data'],time,dico['dist'],opt['slant_stack']['u'],opt)
            if opt['slant_stack']['type']=='vespa':
                dico['vespa'] = self._vespa(dico['data'],time,dico['dist'],opt)
        else:
            dico['data'] = data
            dico['dist'] = dist
        dico['time'] = time
        dico['idvs'] = idvs
        dico['idvr'] = idvr
        dico['lat']  = opt['lat']
        dico['lon']  = opt['lon']
        dico['depth']= opt['depth']
        dico['elev'] = opt['elev']
        dico['id']   = opt['id']
        dico['iddate'] = opt['iddate']
        if opt['iddate'] != -1:
            dico['date1'] = self.md_c['date1'][dico['iddate']]
            dico['date2'] = self.md_c['date2'][dico['iddate']]
        return dico

    def _bin_data(self,data,time,dist,opt) :
        bin_axis = np.arange(min(dist), max(dist)+opt['bin_width'], opt['bin_width'])
        bdata    = np.zeros((len(time),bin_axis.size))
        bn       = np.zeros(bin_axis.size)
        for idx,tr in enumerate(data.transpose()):
            ib = (np.abs(bin_axis-dist[idx])).argmin()
            if not np.isnan(abs(tr).max()):
                bdata[:,ib] += tr
            bn[ib]      += 1
        for idx,tr in enumerate(bdata.transpose()):
            if opt['norm_tr'] and abs(tr).max(): 
                bdata[:,idx]=tr/abs(tr).max()
            elif bn[idx]: bdata[:,idx]=tr/bn[idx]
        return bdata,bin_axis,bn

    def _slant_stack(self,data,time,dist,u,opt) :
        #ipdb.set_trace()
        vdist = dist-np.mean(dist)
        bdata = np.zeros(data.shape)
        range_dist = max(dist)-min(dist)
        if u:
            max_delay  = abs(10*range_dist * u)
            #dist  = np.mean(dist)
            delay    = vdist*u
            delta = np.zeros((int(max_delay/self.md_c['tau']*2+1),data.shape[1]))
            if data.any():
                for idx,tr in enumerate(data.transpose()):
                    delta[np.round(delta.shape[0]/2).astype('int')  - np.round(delay[idx]/self.md_c['tau']).astype('int') ,idx] = 1.
                    bdata[:,idx]=signal.fftconvolve(tr, delta[:,idx], mode='same')
        norm_value = float(len(np.where(np.max(bdata,axis=0))[0]))
        if norm_value == 0. : norm_value =1.
        if opt['slant_stack']['sum']['type'] == 'pws':
            return pws(bdata.transpose(),opt['slant_stack']['sum']['timegate'],
                opt['slant_stack']['sum']['power'],1/self.md_c['tau'],'float32')/norm_value
        else:
            return np.sum(bdata,axis=1)/norm_value
        


    def _vespa(self,data,time,dist,opt) :
        vespa = np.zeros((len(time),len(opt['slant_stack']['u'])))
        d     = np.zeros(data.shape)
        for idx,slow in enumerate(opt['slant_stack']['u']):
            vespa[:,idx] = self._slant_stack(data,time,dist,slow,opt)
        return vespa


    def _find_id(self,idv,internal_id) :
        I1=np.array(())
        I2=np.array(())
        for iidv in idv:
            I1= np.unique(np.hstack((I1,np.where(internal_id[:,0]==iidv)[0])))
            I2= np.unique(np.hstack((I2,np.where(internal_id[:,1]==iidv)[0])))
        return np.unique(np.hstack((I1,I2)))  # I1,I2
    
    def _idv(self,idv=[],inv=None) :
        idv  = list(idv)
        allid = list(self.get_station_list()['id'])
        if idv==[] and inv==None :return [allid[random.randint(0,len(allid)-1)]]
        if idv==[] and inv=='r'  :return allid
        if 'all' in idv: return allid
        new_idv = []
        for iidv in idv:
            if iidv[-1] == '*':
                lgt = len(iidv[:-1])
                for mot in allid:
                    if mot[0:lgt]==iidv[:-1]:
                        new_idv.append(mot)
            else:new_idv.append(iidv)
        return list(new_idv)

    def _get_file_number(self,I_path) :
        ''' return the indice of the file containing the I_path th path:) '''
        I_file=np.where(self.file_I[:,1] >= I_path)[0][0]
        return I_file

    def __init__(self,in_dir) :
        self.in_dir= in_dir; 
        self.c1_file = glob.glob(in_dir+'/xcorr_*h5')
        self.c1_file.sort()

        db_file = self.in_dir+'/db.pkl'
        if os.path.isfile(db_file) :
            db=load_pkl(db_file)
        else : 
            db = build_db_file(self.c1_file)
            save_as_pkl(db,db_file)
        #self.lon = np.array(db['lon'])
        #self.lat = np.array(db['lat'])
        #self.elev= np.array(db['elev'])
        #self.id  = np.array(db['id'])
        #self.id  = db['id']
        self.lon = db['lon']
        self.lat = db['lat']
        self.elev= db['elev']
        self.depth= db['depth']
        self.id  = db['id'].astype('str')
        self.dist= db['dist'].flatten()
        self.az  = db['az'].flatten()
        self.baz = db['baz'].flatten()
        self.md_c= db['md_c']
        self.in_ = db['in_']
        self.file_I = db['file_I']
        self.rtz    = False
        if 'in_rot' in db:
            self.rtz    = True
            self.in_rot = db['in_rot']


#--------------------------------------------------
#
#  FUNCTIONS THAT ARE OUTSIDE THE CLASS : 
#
#----------------------------------------------------
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

def build_db_file(file_list) : 
    dd.dispc('  creating db file','y','b')
    nfile=len(file_list) 
    db={}
    db['lat'] = [] 
    db['lon'] = [] 
    db['elev']= [] 
    db['depth']= [] 
    db['id']  = []
    db['file_I']=np.zeros((nfile,2))
    npath_prev = 0 
    k=0
    for ifile in file_list : 
        ff = h5py.File(ifile,'r')
        db['lat'].extend(ff['md']['lat'][:].tolist())
        db['lon'].extend(ff['md']['lon'][:].tolist())
        db['elev'].extend(ff['md']['elev'][:].tolist())
        db['depth'].extend(ff['md']['depth'][:].tolist())
        db['id'].extend(ff['md']['id'][:].astype('str').tolist())
        db['file_I'][k,:] = [npath_prev, len(db['lat'])-1]
        npath_prev = len(db['lat'])
        ff.close()
        k=k+1
    # converting list into array : 
    db['id']  = np.array(db['id'])
    db['lon'] = np.array(db['lon']) 
    db['lat'] = np.array(db['lat'])
    db['elev']= np.array(db['elev'])
    db['depth']= np.array(db['depth'])
    #computing distance and baz btw each station pair : 
    npath = len(db['lat']) 
    db['dist']= np.zeros((npath,1))
    db['az']  = np.zeros((npath,1))
    db['baz'] = np.zeros((npath,1))

    for ipath in range(0, npath) :
        if db['lon'][ipath][0] and db['lon'][ipath][1]:            
            [dist,az,baz] = gps2dist(db['lat'][ipath][0],db['lon'][ipath][0],db['lat'][ipath][1],db['lon'][ipath][1])
            db['dist'][ipath] = dist/1000.
            db['az'][ipath]   = az 
            db['baz'][ipath]  = baz 
        else:
            db['dist'][ipath] = degrees2kilometers(abs(db['lat'][ipath][0]-db['lat'][ipath][1]))
            db['az'][ipath]   = 0. 
            db['baz'][ipath]  = 0.       

    #reformatting file_I : 
    #calcul des distances, azimuth, baz .... 
    # adding metadata : 
    ff = h5py.File(file_list[0],'r')
    db['md_c']= h5.read_group_as_dict(ff['md_c'])
    db['in_'] = h5.read_group_as_dict(ff['in_'])
    if 'in_rot' in ff:
        db['in_rot'] = h5.read_group_as_dict(ff['in_rot']) 
    return db 

def load_pkl(filename) : 
    dd.dispc('  loading '+filename,'y','b')
    ff=open(filename,'rb') 
    db=pickle.load(ff)
    ff.close() 
    return db 

def save_as_pkl(db,filename) :
    ff=open(filename,'wb')
    pickle.dump(db,ff)
    ff.close()

def mkdir(out_dir) : 
    if os.path.isdir(out_dir) == False :
        os.makedirs(out_dir)
