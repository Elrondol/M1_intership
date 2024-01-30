import numpy as np
import scipy.signal as signal 


def filter(C,p1,p2,dt,npole=2,plot_filter=False) : 

	b,a = signal.butter(npole,(2.*dt/p2,2.*dt/p1),'bandpass',analog=False,output='ba')

	if np.ndim(C)==2 :
		C = signal.filtfilt(b,a,C,axis=1) #padtype=None)
	else :
		C=signal.filtfilt(b,a,C,axis=0) #padtype=None)

	if plot_filter :
		plt.clf()
		w,h = signal.freqs(b,a)
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
	return C 


def stack_linear(C,stack,stack_step) : 
	[ndate, nwf] = np.shape(C)
	if stack > 1 :
		I1= np.arange(0, ndate-stack+1,stack_step,dtype='int64')
		I2=I1+int(stack)-1
		nstack=len(I1) 
		C2=np.zeros((nstack,nwf))
		for istack in np.arange(0,nstack):
			C2[istack,:] = np.sum(C[I1[istack]:I2[istack],:],axis=0)
	else :	
		I1=np.arange(0,ndate,stack_step,dtype='int64') 
		C2=C[I1,:]
		I2=I1
	return (C2,I1,I2)


def norm(C) :
	if np.ndim(C)==2 :
		nwf, nsig=np.shape(C) 

		for isig in np.arange(0,nsig) :
			max_=np.max(np.abs(C[:,isig]))
			if max_ > 0 :
				C[:,isig]=C[:,isig]/max_ 

	elif np.ndim(C)==1 : 
		max_=np.max(np.abs(C))
		if max_ > 0 : 
			C=C/max_ 
	return C

