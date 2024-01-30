################################################
# m00_get_stations.py 
# Pierre Boue (UGA)
# Laurent Stehly (UGA)
# email: pycorr@univ-grenoble-alpes.fr
# June 2019
################################################
"""

"""

import pickle
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
import m_pycorr.mods.lang as lang
import m_pycorr.mods.dd as dd
from obspy.core import UTCDateTime
from obspy.geodetics import locations2degrees
import numpy as np
global plotmap 
plotmap = False
try:
  import matplotlib.pyplot as plt
  from matplotlib.transforms import offset_copy
  import cartopy.crs as ccrs
  from cartopy.io.img_tiles import Stamen
  import cartopy.feature as cfeature
  plotmap = True
except:
  pass


import ipdb




##########################################################################################
##########################################################################################
#######         ###   #####   ###         ##       ####  ##            ###           #####
#######         ###   #####   ###         ##       ####  ##            ##     ############
#######   #########   #####   ###   ########   ##   ###  ######    ######     ############
#######   ##########   ###   ####   ########   ##   ###  ######    #######        ########
#######      #######   ###   ####      #####   ###   ##  ######    ###########       #####
#######   ###########   #   #####   ########   ###   ##  ######    ###############    ####
#######   ###########   #   #####   ########   ####      ######    ##############     ####
#######         ######     ######         ##   ####      ######    #######           #####
##########################################################################################
##########################################################################################



def find_events(input_user={},datacenter ='IRIS',evtfile='events'):
    ''' this function ...
    '''
    in_ = {}   
    in_['starttime']            = UTCDateTime("2018-01-01T00:00:00")
    in_['endtime']              = UTCDateTime("2018-12-31T00:00:00")
    in_['minlatitude']          = None
    in_['maxlatitude']          = None
    in_['minlongitude']         = None
    in_['maxlongitude']         = None
    in_['latitude']             = None
    in_['longitude']            = None
    in_['minradius']            = None
    in_['maxradius']            = None
    in_['mindepth']             = 0.
    in_['maxdepth']             = 1000.
    in_['minmagnitude']         = 6.
    in_['maxmagnitude']         = 10.
    in_['magnitudetype']        = None
    in_['includeallorigins']    = None
    in_['includeallmagnitudes'] = None
    in_['includearrivals']      = None
    in_['eventid']              = None
    in_['limit']                = None
    in_['offset']               = None
    in_['orderby']              = None
    in_['catalog']              = None
    in_['contributor']          = None
    in_['updatedafter']         = None
    in_['filename']             = None
    in_   = lang.merge_options(in_,input_user)

    try:
        clientcat     = Client(datacenter)
        cat           = clientcat.get_events(**in_)
        dd.dispc('get catalog from ' + datacenter + ' -- ' + str(in_['catalog']) ,'g','n')
        ff   = open(evtfile + '.txt','w')
        flat = []
        flon = []
        fdep = []
        fmag = []
        num_ = 0
        for ev in cat:
            mline = []
            mline.append(ev.origins[0].time)
            mline.append(ev.origins[0].latitude)
            mline.append(ev.origins[0].longitude)
            mline.append(ev.origins[0].depth)
            mline.append(ev.magnitudes[0].mag)
            mline.append(ev.magnitudes[0].magnitude_type)
            fline = str(mline[0])
            for mm in mline[1:]: 
                fline = fline + '   ' + str(mm)
            ff.write(fline + '\n')
            flat.append(ev.origins[0].latitude)
            flon.append(ev.origins[0].longitude)
            fdep.append(ev.origins[0].depth)
            fmag.append(ev.magnitudes[0].mag)
            num_ += 1
        ff.close()
        mmap(flon,flat,fdep,fmag,evtfile + '.png',info=str(in_['starttime'].date) + str(in_['endtime'].date))
        dd.dispc('Successful request !','g','n')
        dd.dispc('number of answer(s) : ' + str(num_) ,'c','n')
    except:
        dd.dispc('failed to process this request','r','n')
        pass
        ff.close()

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
#-------------------- STAIONS SUBFUNCTIONSFUNCTIONS ----------------------------------
#---------------------------------------------------------------------------------------


def mmap(lon=[2.39,5.72],lat=[47.08,45.18],depth=[10,10],mag=[5,10],filename='events.png',info='event map'):
    stamen_terrain = Stamen('terrain-background')
    ax = plt.axes(projection=stamen_terrain.crs)
    minlat = -80
    maxlat = 80
    minlon = -180
    maxlon = 180
    dl     = 0.1
    r      = int(locations2degrees(min(lat),min(lon),max(lat),max(lon))*111)
    if r < 5000:
        if min(lat) - dl > minlat: minlat = min(lat) - dl
        if max(lat) + dl < maxlat: maxlat = max(lat) + dl
        if min(lon) - dl > minlon: minlon = min(lon) - dl
        if max(lon) + dl < maxlon: maxlon = max(lon) + dl
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
    ax.add_image(stamen_terrain,res)
    gl = ax.gridlines(draw_labels=True,linewidth=0.5, color='gray', alpha=0.5)
    gl.xlabels_top   = False
    gl.ylabels_right = False
    #sc = plt.scatter(lon, lat, c=np.array(depth)/1000, s=np.array(mag)**2.5,transform=ccrs.Geodetic(),cmap='plasma_r',alpha=0.5)
    sc = plt.scatter(lon, lat, c=np.array(depth)/1000, s=np.array(mag)**2.5,transform=ccrs.PlateCarree(),cmap='plasma_r',alpha=0.5)
    cb = plt.colorbar(sc)
    cb.set_label('Depth (km)')
    plt.savefig(filename,dpi=150, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
    #plt.show()
    plt.close()







