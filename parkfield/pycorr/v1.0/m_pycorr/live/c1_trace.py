
# class contains methods to process cc traces retrieved by 
# pycorr_x.live_md.read_tr

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

class c1_trace :

    def plot(self) : 
        plt.plot(self.time,self.tr)
        plt.xlabel('correlation time [s]')
        plt.title(self.title)
        plt.show()


    def split_ca(self) :
        m0=int(np.floor(len(self.time)/2))
        c1={}
        c1['time']  = self.time[m0:] 
        c1['P']={}
        c1['P']['trace'] = self.tr[m0:]
        c1['N'] ={} 
        c1['N']['trace'] = np.flipud(self.tr[0:m0+1])
        c1['S']={}
        c1['S']['trace'] = self.tr[m0:] + np.flipud(self.tr[0:m0+1])
        #c1['neg']['time']  = np.flipud(self.time[1:m0+1])
        return c1 


    def filter(self,p1,p2,npole=2,plot_filter=False) :
        '''filter the trace btw p1 and p2 s'''
        b,a = signal.butter(npole,(2.*self.tau/p2,2.*self.tau/p1),'bandpass',analog=False,output='ba')
        pad = signal.tukey(len(self.tr), alpha=0.03)
        self.tr = signal.filtfilt(b,a,self.tr*pad,axis=0) #padtype=None)
        if plot_filter :
            plt.clf()
            w, h = signal.freqs(b,a)
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


    def get_window(self,p2,varargin={}) :
        """ INPUT : 
            - p2 : periods of the signal
              there is no filter applied, but p1 and p2 are used to determine the offset btw sw and codaw.

            OUTPUT : 
            - indice (not time!) of the beginning of the coda/sw in the positive/negative cc time

            NOTE : 
            Le cas des trajets longs n est pas parfaitement gere :
            On fait en sorte que la coda ne puisse pas aller au dela de nwf-5 
            par contre dans ce cas on verifie pas que le debut est avant nwf-5
            => peut induire des crash dans les codes suivants si nwf est trop petit 
            par rapport a la distance inter-station. 
        """

        in_={}
        in_['sw_v1']  = 1.5   # Start of SW window [km/s]',...
        in_['sw_v2']  = 5.    # end of SW window [km/s];
        in_['hwidth'] = 2.    # half width of surface wave  [n*p2]
        in_['coda_dp']= 5.    # How many period after SW window, coda start [n*p2]    
        in_['coda_length'] = 500 #[s]
        in_['plot']   =False  # plot window ?

        in_=lang.parse_options(in_,varargin)

        #get convenient variables here : 
        dt = self.tau 
        nwf=len(self.tr)
        hwidth_npts  = np.round(in_['hwidth']*p2/dt)
        coda_dp_npts = np.round(in_['coda_dp']*p2/dt)

        #defining times where we look for surface waves
        t1=self.dist/in_['sw_v2'];
        t2=self.dist/in_['sw_v1'];
        if t2-t1 < dt    : # can happend for close stations
            t2 = t1+5*dt     # subjective
        I0= np.where(self.time >=0)[0][0]

        #preparing output : 
        win       ={}
        win['pos']={}
        win['neg']={}
        win['success']=True 
        # si la distance trop grande => sw hors de la correlation :
        if t1 > self.time[-1] :
            win['success']=False 
            return win 
        #analysing positive correlation time #on se laisse une marge de 5 pts :
        win['pos']['sw_guess']  = [np.nonzero(self.time >=t1)[0][0], np.nonzero(self.time <=t2)[0][-1] ]
        sw_I = np.argmax(abs(self.tr[win['pos']['sw_guess'][0]:win['pos']['sw_guess'][1]])) 
        win['pos']['sw_center']= int(sw_I+win['pos']['sw_guess'][0])
        win['pos']['sw']=[]
        win['pos']['sw'].append(int(max((I0,win['pos']['sw_center'] - hwidth_npts))))
        win['pos']['sw'].append(int(min((nwf,win['pos']['sw_center']+hwidth_npts))))
        win['pos']['coda']=[]
        win['pos']['coda'].append(int(min((nwf-5,win['pos']['sw_center']+coda_dp_npts))))
        win['pos']['coda'].append(int(min(np.round(win['pos']['coda'][0]+in_['coda_length']/dt),nwf-1)))

        #analysing negative correlation time : 
        win['neg']['sw_guess']  = [np.nonzero(self.time >= -t2)[0][0],np.nonzero(self.time <= -t1)[0][-1]]
        sw_I=int(np.argmax(abs(self.tr[win['neg']['sw_guess'][0]:win['neg']['sw_guess'][1]])))
        win['neg']['sw_center']=int(sw_I + win['neg']['sw_guess'][0])
        win['neg']['sw']=[]
        win['neg']['sw'].append(int(max((1,win['neg']['sw_center'] - hwidth_npts))))
        win['neg']['sw'].append(int(min((I0,win['neg']['sw_center']+hwidth_npts))))
        win['neg']['coda']=[] 
        win['neg']['coda'].append(1) 
        win['neg']['coda'].append(int(max((5,win['neg']['sw_center']- coda_dp_npts))))
        win['neg']['coda'][0] = int(max(1,win['neg']['coda'][1]-in_['coda_length']/dt))

        #put them as absolute time for c3 computation : 
        win['coda_start']=np.abs(self.time[[win['pos']['coda'][0], win['neg']['coda'][1]]])
        win['coda_end']  =np.abs(self.time[[win['pos']['coda'][1],win['neg']['coda'][0]]])
        win['sw_start']  =np.abs(self.time[[win['pos']['sw'][0], win['neg']['sw'][1]]])
        win['sw_end']    =np.abs(self.time[[win['pos']['sw'][1],win['neg']['sw'][0]]])

        if in_['plot'] == True : 
            fig,ax=plt.subplots()
            plt.plot(self.time,s2d.norm(self.tr),color='k')
            for ica in ['pos','neg'] : 
                plt.axvline(self.time[win[ica]['sw_center']],color='red',lw=2)
                plt.axvline(self.time[win[ica]['sw'][0]]  ,  color='blue')
                plt.axvline(self.time[win[ica]['sw'][1]]  ,  color='blue')
                plt.axvline(self.time[win[ica]['coda'][0]],  color='yellow')
                plt.axvline(self.time[win[ica]['coda'][1]],  color='yellow')
                xlimit=np.round(self.dist*20*dt)
                plt.xlim((-xlimit,xlimit))
                plt.ylim((-1.2,1.2))

                dx= self.time[win[ica]['sw_guess'][1]] - self.time[win[ica]['sw_guess'][0]]
                ax.add_patch(pt.Rectangle((self.time[win[ica]['sw_guess'][0]],-1.2),dx,0.2,color='red',fill=True))

                dx= self.time[win[ica]['sw'][1]] - self.time[win[ica]['sw'][0]]
                ax.add_patch(pt.Rectangle((self.time[win[ica]['sw'][0]],-1),dx,2,color='blue',fill=True,alpha=0.3))

                dx= self.time[win[ica]['coda'][1]] - self.time[win[ica]['coda'][0]]
                ax.add_patch(pt.Rectangle((self.time[win[ica]['coda'][0]],-0.5),dx,1,color='yellow',fill=True,alpha=0.3))
            plt.show()
        return win 

    def __init__(self,tr=[]) : 
        ''' if tr = empty list => return an empty object = no data                otherwise if tr is a dict: fill the object with tr keys 
            tr is the output of pycorr_x.live_md.read_c1 '''
        if len(tr)==0 : 
            return 
        else      :
            for ikey in tr.keys() : 
                self.__dict__[ikey] = tr[ikey]
