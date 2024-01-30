import sys,os,glob,time,copy,warnings,cProfile,shutil,matplotlib
from IPython import display
import h5py as h5
import numpy as np
import scipy.fftpack
import scipy.signal
import scipy.io as io
import skimage.restoration as deconv
from math import cos, sin, radians
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42


import m_beam
import io_beam
import plt_beam
import dd
import lang

from functions import * ###################################### rajouter des fonctions par ex pour calculer la posiiton du barycentre
from variables import *

import ipdb

plt.close('all')

plot_day = tt['day']  # 'stack' # int or tt['day'] or 'stack'
ref_day  = 300  # ref day to take 'stack' case. for instance if the array slightly change during the experiment .. use the ARF of this day
num_colu = 3 # number of column in the subplot ...


if os.path.isdir(pl_updated['save_fig'])==False :
    os.makedirs(pl_updated['save_fig'])

basename  = '/beam_' +  compo_to_plot
basename2 = in_['beam_dir'] + basename
    
if plot_day == 'stack':
    beam_file_s   = basename2 + '_stack.h5'
    init = True
    for index_day in tt['day']:
        if use_pre_stacked:
            for name in [glob.glob(basename2 + '_t*' + '_day_%03d.h5'%index_day)[index] for index in [0,3,4]]:
                beam_file = name
                print(beam_file)
                h5b   = h5.File(beam_file,'r')
                if init:
                    beam_file_ref = beam_file
                    shutil.copyfile(beam_file_ref,beam_file_s)
                    h5s   = h5.File(beam_file_s,'a')         
                fks   = h5s['fk']       # load the data
                if init:
                    fk  = np.zeros(fks.shape,dtype=np.complex128)
                    init = False
                else:
                    fk = fks[()]
                print('stack ABS(FK) ...')
                fks[...] = fk + np.abs(h5b['fk'][()]/len(tt['day']))
                h5b.close()
        else :
            beam_file = basename2 + '_day_%03d.h5'%index_day
            print(beam_file)
            h5b   = h5.File(beam_file,'r')
            fks   = h5s['fk']       # load the data
            if init:
                fk  = np.zeros(fks.shape,dtype=np.complex128)
                init = False
            else:
                fk = fks[()]
            print('stack ABS(FK) ...')
            fks[...] = fk + np.abs(h5b['fk'][()]/len(tt['day']))
            h5b.close()
    h5s.close()
    plt_beam.make_subfig(beam_file_s,pl_arg=pl_updated,num_colu=num_colu,info=compo_to_plot + '_day_stack')
    fig = plt.gcf()
    fig.savefig(pl_updated['save_fig'] + basename + f'_stack.{pl_updated["format"]}',format=pl_updated['format'])
elif isinstance(plot_day,list):
    for iday in plot_day:        
        if use_pre_stacked:
            for name in glob.glob(basename2 + '_t*' + '_day_%03d.h5'%iday):
                beam_file = name
                baname    = name.split('/')[-1].split('.')[0].split('beam_')[-1]
                fig = plt.gcf()
                plt_beam.make_subfig(beam_file,pl_arg=pl_updated,num_colu=num_colu,info=baname)
                fig = plt.gcf()
                fig.savefig(pl_updated['save_fig'] + basename + '%03d'%iday + baname + f".{pl_updated['format']}" ,format=pl_updated['format'])
                display.clear_output(wait=True)
                display.display(fig)
                #wait = input("PRESS ENTER TO CONTINUE.")
                plt.close()
        else:    
            beam_file = basename2 + '_day_%03d.h5'%iday
            fig = plt.gcf()
            plt_beam.make_subfig(beam_file,pl_arg=pl_updated,num_colu=num_colu,info=compo_to_plot + '_day_%03d'%iday)
            fig = plt.gcf()
            fig.savefig(pl_updated['save_fig'] + basename + f"_day_%03d.{pl_updated['format']}"%iday,format=pl_updated['format'])
            display.clear_output(wait=True)
            display.display(fig)
            #wait = input("PRESS ENTER TO CONTINUE.")
            plt.close()
else:
    beam_file = basename2 + '_day_%03d.h5'%plot_day
    fig = plt.gcf()
    plt_beam.make_subfig(beam_file,pl_arg=pl_updated,num_colu=num_colu,info=compo_to_plot + '_day_%03d'%plot_day)
    fig = plt.gcf()
    fig.savefig(pl_updated['save_fig'] + basename + f"_day_%03d.{pl_updated['format']}"%plot_day,format=pl_updated['format'])
