import numpy as np
import pandas as pd
from pyproj import Transformer

import matplotlib.pyplot as plt

from datetime import datetime, timedelta


from tqdm import tqdm
import h5py

from sklearn.cluster import DBSCAN
import scipy.ndimage
import scipy
import matplotlib.cm as cm
from skimage.feature import blob_doh, blob_dog, blob_log
from findpeaks import findpeaks

from matplotlib.transforms import Bbox

from obspy.geodetics.base import gps2dist_azimuth
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.img_tiles import Stamen, OSM, GoogleTiles
from matplotlib_scalebar.scalebar import ScaleBar
from geopy.distance import distance, geodesic


from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from obspy.core.utcdatetime import UTCDateTime
from scipy.signal import hilbert,find_peaks, butter, filtfilt
from matplotlib.collections import LineCollection

from typing import List



def extract_coordinates(path,sta_name=False,sta_loc=False):
    '''Fonction qui récupère en input la liste des station et qui va  extraire les coordonnées
    L'option sta_name permet de récupérer les noms des stations en plus des autres listes'''
    #laoding the station_list, cette version de station list devra être la version avec les stations non utilisées virées 
    latitudes = pd.read_csv(path, header=None, sep='    ',usecols=[4],engine='python').values
    longitudes = pd.read_csv(path, header=None, sep='    ', usecols=[5],engine='python').values
    heights = pd.read_csv(path, header=None, sep='    ', usecols=[6], engine='python').values
    latitudes_clean = []
    longitudes_clean = []
    heights_clean = []
    if sta_name==True:
        networks = pd.read_csv(path, header=None, sep='    ',usecols=[1],engine='python').values
        networks_clean = []
        stations = pd.read_csv(path, header=None, sep='    ',usecols=[2],engine='python').values
        stations_clean = []
    
    if sta_loc==True:
        locations = pd.read_csv(path, header=None, sep='    ',usecols=[3],engine='python').values
        locations_clean = []
        
    for i in range(len(latitudes)):
        latitudes_clean.append(latitudes[i][0])
        longitudes_clean.append(longitudes[i][0])
        heights_clean.append(heights[i][0])
        if sta_name==True:
            networks_clean.append(networks[i][0])
            stations_clean.append(stations[i][0])
        if sta_loc==True:
            if locations[i][0]=='--':
                locations_clean.append('00')
            else:
                locations_clean.append(locations[i][0])
        
    latitudes_clean = np.array(latitudes_clean)    
    longitudes_clean = np.array(longitudes_clean)
    heights_clean = np.array(heights_clean)

    if sta_name == True and sta_loc ==True:
        networks_clean = np.array(networks_clean)
        stations_clean = np.array(stations_clean)
        locations_clean = np.array(locations_clean)
        return longitudes_clean,latitudes_clean,heights_clean, networks_clean, stations_clean, locations_clean
    elif sta_name == True:
        networks_clean = np.array(networks_clean)
        stations_clean = np.array(stations_clean)
        return longitudes_clean,latitudes_clean,heights_clean, networks_clean, stations_clean
    elif sta_loc ==True:
        locations_clean = np.array(locations_clean)
        return longitudes_clean,latitudes_clean,heights_clean, locations_clean
    else:
        return longitudes_clean,latitudes_clean,heights_clean

def compute_barycenter(path):
    '''Fonction permettant de calculer les coordonnées du barycentre de de l'array à partir du fichier station list'''
    longitudes_clean,latitudes_clean,heights_clean = extract_coordinates(path)
    #loading the transformers to convert coordinates to     
    trans_GPS_to_XYZ = Transformer.from_crs(4979, 4978, always_xy=True)
    trans_XYZ_to_GPS = Transformer.from_crs(4978,4979,always_xy=True)
    #on a créé les convertisseurs    
    X,Y,Z = trans_GPS_to_XYZ.transform(longitudes_clean, latitudes_clean, heights_clean)
    X_bary = np.mean(X) 
    Y_bary = np.mean(Y)
    Z_bary = np.mean(Z)
    lon_bary,lat_bary,height_bary = trans_XYZ_to_GPS.transform(X_bary,Y_bary,Z_bary)
    return lon_bary, lat_bary, height_bary



def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc
        
    
    
def read_datetime_list_from_file(file_path: str) -> List[datetime]:
    datetime_list = []
    with open(file_path, 'r') as file:
        for line in file:
            dt_str = line.strip()
            dt = datetime.fromisoformat(dt_str)
            datetime_list.append(dt)
    return datetime_list
