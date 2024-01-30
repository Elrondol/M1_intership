################################################
# m00_get_stations.py 
# Pierre Boue (UGA)
# Laurent Stehly (UGA)
# email: pycorr@univ-grenoble-alpes.fr
# May 2017
################################################
"""

"""

import pickle
import urllib
#import urllib2
import numpy as np 
import m_pycorr.mods.lang as lang
import m_pycorr.mods.dd as dd
from obspy.core import UTCDateTime
from obspy.geodetics import locations2degrees

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
##          ##             #####    #####             ##    ##          ##       ####   ##
#    #########             ####      ####             ##    ##          ##       ####   ##
#    ##############   ########        ########   #######    ##   ####   ##   ##   ###   ##
##       ##########   #######    ##    #######   #######    ##   ####   ##   ##   ###   ##
######      #######   ######    ####    ######   #######    ##   ####   ##   ###   ##   ##
##########   ######   #####              #####   #######    ##   ####   ##   ###   ##   ##
##########   ######   ####    ########    ####   #######    ##          ##   ####       ##
##          #######   ###    ##########    ###   #######    ##          ##   ####       ##
##########################################################################################
##########################################################################################


def find_stations(input_user={},stafile='stations',all_locid_and_channel=False):
    ''' this function check the data availability on FDSN database 
    It creates the *stafile*.txt file that is needed for following processing
    all stations metadata are also saved in *stafile*.pkl as a large dictionary 
    If cartopy and matplotlib are availble, maps are also created (*stafile*.pdf)
    '''

    values                    = {}
    values['net']             = 'FR,GR'
    values['sta']             = '*'
    values['loc']             = '*'
    values['channel']         = 'BH*'
    values['format']          = 'text'
    values['includeoverlaps'] = 'false'
    values['nodata']          = '404'
    values   = lang.merge_options(values,input_user)
    url      = 'http://service.iris.edu/irisws/fedcatalog/1/query?'
    data     = urllib.parse.urlencode(values)
    try:
        f    = urllib.request.urlopen(url + data)
        p    = f.read().decode('utf-8').split('\n')

        idico  = {}
        dico   = {}

        inc = 0
        for kk in p[0].split(' | '):
            if kk[0]=='#': kk=kk[1:]
            idico["%03d"%inc] = kk
            inc+=1
        dc = []

        for lll in p[2:]:
            if lll:
                if lll[0:11]=='#DATACENTER':
                    dc = lll.split(',')[0].split('=')[-1]
                    if dc=='IRISDMC' :dc = 'IRIS'
                    if dc=='SED'     :dc = 'ETH'
                    if dc=='GEOFON'  :dc = 'GFZ'
                    if dc=='USPSC'   :dc = 'USP'
                    dico[dc] = {}
                else:
                    inc          = 0
                    line_info    = lll.split('|')
                    short_Kname  = line_info[0] + '_' + line_info[1]
                    Kname        = line_info[0] + '_' + line_info[1] + '_' + line_info[2] + '_' + line_info[3] 
                    if Kname not in dico[dc]:
                        dico[dc][Kname] = {}
                        for vv in line_info:
                            if idico["%03d"%inc] in ['StartTime','EndTime']:
                                dico[dc][Kname][idico["%03d"%inc]] = UTCDateTime(vv)
                            elif idico["%03d"%inc] in ['Azimuth','Depth','Dip','Elevation','Latitude','Longitude','SampleRate']:
                                if vv=='':
                                    dico[dc][Kname][idico["%03d"%inc]] = 0.
                                else: 
                                    dico[dc][Kname][idico["%03d"%inc]] = float(vv)
                            else :
                                dico[dc][Kname][idico["%03d"%inc]] = vv
                            inc+=1
                    else:
                        if UTCDateTime(line_info[int(list(idico.keys())[list(idico.values()).index('StartTime')])]) < dico[dc][Kname]['StartTime']:
                            dico[dc][Kname]['StartTime'] = UTCDateTime(line_info[int(list(idico.keys())[list(idico.values()).index('StartTime')])])
                        if UTCDateTime(line_info[int(list(idico.keys())[list(idico.values()).index('EndTime')])]) > dico[dc][Kname]['EndTime']:
                            dico[dc][Kname]['EndTime'] = UTCDateTime(line_info[int(list(idico.keys())[list(idico.values()).index('EndTime')])])

        # keep only the first Kname (alphabetical order)
        list_of_sKname = []
        list_of_Kname  = []
        if not all_locid_and_channel:
            sdico = {}
            for dc in dico.keys():
                sdico[dc] = {}
                for Kname in np.sort([*dico[dc].keys()]).tolist():
                    #list_of_Kname.append(Kname + ' from' + dc)
                    sKname = dico[dc][Kname]['Network'] + '_' + dico[dc][Kname]['Station']
                    if sKname not in list_of_sKname:
                        list_of_sKname.append(sKname)
                        list_of_Kname.append(Kname + ' from ' + dc)
                        sdico[dc][sKname] = dico[dc][Kname]
                    else:
                        firstKname =   list_of_Kname[list_of_sKname.index(sKname)]
                        #dd.dispc('multiple answer for ' + sKname + ' keeping ' + firstKname,'y','n')
            dico = sdico

        dd.dd(dico)
        ff = open(stafile + '.pkl','wb')
        ff.close()
        num_ = 0
        ff = open(stafile + '.txt','w')
        if not all_locid_and_channel: 
            list_file = ['DC','Network','Station','Location','Latitude','Longitude','Elevation','Depth']
        else: 
            list_file = ['DC','Network','Station','Location','Channel','Latitude','Longitude','Elevation','Depth']
            dd.dispc('Channel id not yet supported in station.txt','r','n')
        for dc in dico.keys():
            for Kname in dico[dc]:
                mline = []
                for kk in list_file:
                    if kk == 'DC': 
                        mline.append(dc)
                    else:
                        if str(dico[dc][Kname][kk]): 
                            mline.append(str(dico[dc][Kname][kk]))
                        else: 
                            mline.append('--')
                fline = mline[0]
                for mm in mline[1:]: 
                    fline = fline + '    ' + mm
                num_ += 1
                ff.write(fline + '\n')
        ff.close()

        if plotmap:
            flat = []
            flon = []
            for dc in dico:
                lat = []
                lon = []
                for kname in dico[dc]:
                    lat.append(dico[dc][kname]['Latitude'])
                    lon.append(dico[dc][kname]['Longitude'])
                #mmap(lon,lat,dc + '_stations.pdf')
                flat = flat+lat
                flon = flon+lon
            mmap(flon,flat,stafile + '.png',info=url+data)
        dd.dispc('Successful request ! \n' + url + data,'g','n')
        dd.dispc('number of answer(s) : ' + str(num_) ,'c','n')
    except:
        dd.dispc('failed to process this request : ' + url + data,'r','n')
        pass


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


def mmap(lon=[2.39,5.72],lat=[47.08,45.18],filename='stations.png',info='station map'):
    stamen_terrain = Stamen('terrain-background')
    ax = plt.axes(projection=stamen_terrain.crs)
    minlat = -80
    maxlat = 80
    minlon = -180
    maxlon = 180
    dl     = 0.1
    r      = int(locations2degrees(min(lat),min(lon),max(lat),max(lon))*111)
    if r < 10000:
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
    #plt.scatter(lon, lat, c='red', s=20,transform=ccrs.Geodetic(),cmap='plasma_r',alpha=0.5)
    plt.scatter(lon, lat, c='red', s=20,transform=ccrs.PlateCarree(),cmap='plasma_r',alpha=0.5)
    plt.savefig(filename,dpi=150, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
    #plt.show()
    plt.close()







