# This script will list events from fdsn webservices
# Output is a list as a light .txt files, and a  map .png
# 
# OUTPUT file structure :
# event_id                      lat      lon        depth      mag   mag_type
# 2010-07-23T22:51:13.000000Z   6.4232   123.5796   584700.0   7.7   MW
#
########################################################
######################################################## 
########################################################  
import m_pycorr.m01_get_events as get_ev
from obspy.core import UTCDateTime

# arguments are same as obspy.clients.fdsn.client.Client.get_events

datacenter ='IRIS' # datacenter for request
evtfile    ='events' # output name


in_ = {}
in_['starttime']            = UTCDateTime("2001-12-31T12:00:00")
in_['endtime']              = UTCDateTime("2002-01-01T00:00:00")
in_['minlatitude']          = None
in_['maxlatitude']          = None
in_['minlongitude']         = None
in_['maxlongitude']         = None
in_['latitude']             = None
in_['longitude']            = None
in_['minradius']            = None
in_['maxradius']            = None
in_['mindepth']             = 50.
in_['maxdepth']             = 1000.
in_['minmagnitude']         = 7.5
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


get_ev.find_events(in_,datacenter =datacenter,evtfile=evtfile)
