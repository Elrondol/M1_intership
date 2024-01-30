
# class contains methods to process cc traces retrieved by 
# m_live_md.read_tr

import numpy as np
import scipy.signal as signal 
import m_pycorr.live.mods.s2d as s2d 
import matplotlib.pyplot as plt
import matplotlib.patches as pt 
from matplotlib.collections import PatchCollection

import m_pycorr.mods.lang as lang 
try : 
	import ipdb
except :
	pass 

class data_trace :

	def plot(self) : 
		plt.plot(self.time,self.tr)
		plt.xlabel('correlation time [s]')
		plt.title(self.title)
		plt.show()

	def split_ca(self) :
		m0=int(np.floor(len(self.time)/2))
		c1={}
		c1['time']  = self.time[m0:-1] 
		c1['P']={}
		c1['P']['trace'] = self.tr[m0:-1]
		c1['N'] ={} 
		c1['N']['trace'] = np.flipud(self.tr[1:m0+1])
		#c1['neg']['time']  = np.flipud(self.time[1:m0+1])
		return c1 


	def filter(self,p1,p2,npole=2,plot_filter=False) :
		'''filter the trace btw p1 and p2 s'''
		b,a = signal.butter(npole,(2.*self.tau/p2,2.*self.tau/p1),'bandpass',analog=False,output='ba')
		self.tr = signal.filtfilt(b,a,self.tr,axis=0) #padtype=None)
		if plot_filter :
			plt.clf()
		 	w, h = signal.freqs(b, a)
		 	plt.plot(2*3.14/w, 20 * np.log10(abs(h)))
			plt.title('Butterworth filter frequency response')
			plt.xlabel('periods [second]')
			plt.ylabel('Amplitude [dB]')
			plt.margins(0, 0.1)
			plt.grid(which='both', axis='both')
			plt.axvline(p1, color='blue') # cutoff frequency
			plt.axvline(p2, color='red') # cutoff frequency
			plt.xlim((0,100))
			plt.show()
 

	def __init__(self,tr=[]) : 
		''' if tr = empty list => return an empty object = no data   			 otherwise if tr is a dict: fill the object with tr keys 
			tr is the output of m_live.data_md.read_data '''
		if len(tr)==0 : 
			return 
		else 	 :
			for ikey in tr.keys() : 
				self.__dict__[ikey] = tr[ikey]
