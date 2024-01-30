import os , glob , h5py, cPickle
import numpy as np 
import m_pycorr.mods.dd as dd 
import m_pycorr.live.mods.h5 as h5 
from m_pycorr.live.data_trace import data_trace as data_trace  
from obspy.geodetics import gps2dist_azimuth as gps2dist 
try :
	import ipdb 
except:
	pass 


#-----------------------------------------------------
# Typical usage : 
# import m_live.data_md as data_live
# h1 = data_live('data_1.0Hz')
# h1.get_station_list()
# [...]
#
# -------------------------------------------
# READING methods : 

#---------------------------------------------------------------------------------
#
# UTILITY methods :
# 
# get_station_list : return a dict containing the metadata of all stations
#----------------------------------------------------------------------------------------- 
#
# CONSTRUCTOR : 
#
#-----------------------------------------------------------------------------------------------------


class data_md : 
	'''
	When run the 1st time, it scans all correlations and create a db.pkl file  containing their metadata. 
	'''

	def read_data_for_path_between_date(self,id1,id2,date1,date2,I_cmp=0) :
		''' read all data between id1 and id2 at all date between date1 and date2, and sum them.
	      if the data id1-id2 do not exist, read id2-id1 and time reverse it 
	      id1, id2    : id of both station 
	      date1,date2 : UTCDatetime date
	      I_cmp       : indice of the components [ZZ,ZN,...] to be read 

		'''
		nwf   = len(self.md_c['t'])
		data_tr = np.zeros((nwf))
		#0. get indice of all correlations starting after date1 and ending before date2 
		I = np.intersect1d(np.where(self.md_c['date1']>=date1), np.where(self.md_c['date2']<=date2))
		#1. read them all and sum them :
		for I_date in I : 
					tr, is_data = self.read_data_for_path(id1,id2,I_cmp=I_cmp,I_date=I_date)
					if is_data : 
						data_tr += tr.tr
		# read the reference to get the metadata and substuting the trace by data_tr: 
		data, is_data = self.read_data_for_path(id1,id2,I_cmp=I_cmp,I_date=-1)
		if is_data :
			data.tr = data_tr 
			return data, True 
		else : 
			return [], False 

	def read_data_for_path(self,id1,id2,I_cmp=0,I_date=-1) : 
		''' read data between id1 and id2. If the data id1-id2 does not exist it read the data id2-id1 	 and time reverse it.
			id1 = CH.TORNY.00 
			id2 = CH.SLE.00 
			return an empty dict if data was not found 
		'''
		I1=np.intersect1d(np.where(self.id[:,0]==id1), np.where(self.id[:,1]==id2))
		I2=np.intersect1d(np.where(self.id[:,0]==id2), np.where(self.id[:,1]==id1))
		if len(I1) > 0 :
			I=I1[0] 
			reverse = False  
			data_found= True 
		elif len(I2) > 0 :
			I=I2[0]
			reverse = True
			data_found = True 
		else : 
			data_found = False 

		if data_found :
			tr=self.read_data(I,I_cmp=I_cmp,I_date=I_date,reverse=reverse) 
			return tr,True 
		else : 
			return [],False


	def read_data(self,I_path,I_cmp=0,I_date=-1,reverse=False)  : 
		''' read a single correlate 
		    I_path : indice of the path 
		    I_cmp  : indice of the component 
			I_date : indice of the date. -1 = reference 
			reverse: time reverse the correlations and metadata 
 					 (useful for c3 computation)
		'''
		ccmp = self.in_['cc_cmp'][I_cmp]

		I_file = self._get_file_number(I_path)
		ff = h5py.File(self.data_file[I_file])
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
		tr['file']  = self.data_file[I_file] 
		tr['time']  = self.md_c['t']
		tr['tau']   = self.md_c['tau']
		return data_trace(tr)

	def get_station_list(self) : 
		sta={}
		[sta['id'], I]=np.unique(self.id.flatten(),return_index=True)
		sta['lat']    =self.lat.flatten()[I] 
		sta['lon']    =self.lon.flatten()[I] 
		sta['elev']   =self.elev.flatten()[I]
		return sta 


	def _get_file_number(self,I_path) : 
		''' return the indice of the file containing the I_path th path:) '''
		I_file=np.where(self.file_I[:,1] >= I_path)[0][0]
		return I_file

	def __init__(self,in_dir) : 
		self.in_dir= in_dir; 
		self.data_file = glob.glob(in_dir+'/xcorr_*h5')
		self.data_file.sort()

		db_file = self.in_dir+'/db.pkl'
		if os.path.isfile(db_file) :
			db=load_pkl(db_file)
		else : 
			db = build_db_file(self.data_file)
			save_as_pkl(db,db_file)
		#self.lon = np.array(db['lon'])
		#self.lat = np.array(db['lat'])
		#self.elev= np.array(db['elev'])
		#self.id  = np.array(db['id'])
		#self.id  = db['id']
		self.lon = db['lon']
		self.lat = db['lat']
		self.elev= db['elev']
		self.id  = db['id']
		self.dist= db['dist']
		self.az  = db['az']
		self.baz = db['baz']
		self.md_c= db['md_c']
		self.in_ = db['in_']
		self.file_I = db['file_I']


#--------------------------------------------------
#
#  FUNCTIONS THAT ARE OUTSIDE THE CLASS : 
#
#----------------------------------------------------
def build_db_file(file_list) : 
	dd.dispc('  creating db file','y','b')
	nfile=len(file_list) 
	db={}
	db['lat'] = [] 
	db['lon'] = [] 
	db['elev']= [] 
	db['id']  = []
	db['file_I']=np.zeros((nfile,2))
	npath_prev = 0 
	k=0
	for ifile in file_list : 
		ff = h5py.File(ifile,'r')
		db['lat'].extend(ff['md']['lat'][:].tolist())
		db['lon'].extend(ff['md']['lon'][:].tolist())
		db['elev'].extend(ff['md']['elev'][:].tolist())
		db['id'].extend(ff['md']['id'][:].tolist())
		db['file_I'][k,:] = [npath_prev, len(db['lat'])-1]
		npath_prev = len(db['lat'])
		ff.close()
		k=k+1
	# converting list into array : 
	db['id']  = np.array(db['id'])
	db['lon'] = np.array(db['lon']) 
	db['lat'] = np.array(db['lat'])
	db['elev']= np.array(db['elev'])

	#computing distance and baz btw each station pair : 
	npath = len(db['lat']) 
	db['dist']= np.zeros((npath,1))
	db['az']  = np.zeros((npath,1))
	db['baz'] = np.zeros((npath,1))
	for ipath in range(0, npath) :
		[dist,az,baz] = gps2dist(db['lat'][ipath][0],db['lon'][ipath][0],db['lat'][ipath][1],db['lon'][ipath][1])
		db['dist'][ipath] = dist/1000. 
		db['az'][ipath]   = az 
		db['baz'][ipath]  = baz 

	
	
	#reformatting file_I : 

	#calcul des distances, azimuth, baz .... 


	# adding metadata : 
	ff = h5py.File(file_list[0])
	db['md_c']= h5.read_group_as_dict(ff['md_c'])
	db['in_'] = h5.read_group_as_dict(ff['in_']) 
	return db 

def load_pkl(filename) : 
	dd.dispc('  loading '+filename,'y','b')
	ff=open(filename,'r') 
	db=cPickle.load(ff)
	ff.close() 
	return db 

def save_as_pkl(db,filename) :
	ff=open(filename,'w')
	cPickle.dump(db,ff)
	ff.close()
