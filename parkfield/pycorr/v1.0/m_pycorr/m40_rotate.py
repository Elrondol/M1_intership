################################################
# m40_rotate.py 
# Pierre Boue (UGA)
# Laurent Stehly (UGA)
# email: pycorr@univ-grenoble-alpes.fr
# November 2017
################################################


import sys
import os 
import glob
import time
import copy
import pickle
import h5py as h5 
import numpy as np 
import scipy.io as io
import multiprocessing
from obspy.core import UTCDateTime, Stream
from obspy.geodetics import locations2degrees
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.detrend import polynomial
from obspy.core.inventory.inventory import read_inventory
from obspy import read

import m_pycorr.mods.lang as lang
import m_pycorr.mods.dd as dd

try : 
    import ipdb 
except :
    pass 

#########################################################################################
#########################################################################################
########    ##########            ######            #####            ####################
########    #########     ####     ####     ####     ####    ####     ###################
########    #########    ######    ####    ######    ####    #####    ###################
########    #########    ######    ####    ######    ####    ####     ###################
########    #########    ######    ####    ######    ####           #####################
########    #########    ######    ####    ######    ####    ############################
########          ###     ####     ####     ####     ####    ############################
########          ####            ######            #####    ############################
#########################################################################################
#########################################################################################

def main_rot(input_user={}):
    in_ = {}
    in_['path']      = 'C1_04_xcorr_test__xcorr_norm/'
    in_['mode']      = 'ref_only' # 'all' (cc + ref)
    in_['cc_cmp']    = ['ZZ','RR','TT'] #...
    in_['format']    = 'float16' #...
    in_              = lang.parse_options(in_,input_user)

    list_file     = glob.glob(in_['path']+'/*h5')
    list_file_rtz = glob.glob(in_['path']+'/*_RTZ.h5')
    for del_f in list_file_rtz:list_file.remove(del_f)


    compos         = ['EE','EN','EZ','NE','NN','NZ','ZE','ZN','ZZ']
    new_compos     = ['RR','RT','RZ','TR','TT','TZ','ZR','ZT','ZZ']
    requested      = np.array([[1,1,0,1,1,0,0,0,0], 
                      [1,1,0,1,1,0,0,0,0],
                      [0,0,1,0,0,1,0,0,0],
                      [1,1,0,1,1,0,0,0,0],
                      [1,1,0,1,1,0,0,0,0],
                      [0,0,1,0,0,1,0,0,0],
                      [0,0,0,0,0,0,1,1,0],
                      [0,0,0,0,0,0,1,1,0],
                      [0,0,0,0,0,0,0,0,1]])

    start    = time.clock()
    for file_in in list_file:
        outdir = os.path.dirname(file_in) + '/RTZ/'
        mkdir(outdir)
        file_out  = outdir + os.path.basename(file_in[:-3]) + '_RTZ.h5'
        lock_file = outdir + os.path.basename(file_in[:-3]) + '_RTZ.lock'
        if os.path.isfile(lock_file) or os.path.isfile(file_out):
            dd.dispc(file_out + " and/or lock_file already exist",'r','n')
            continue
        create_lock_file(lock_file)

        fin  = h5.File(file_in, "r")
        fout = h5.File(file_out, "a")
        
        if '/md' in fin: fout.copy(fin['/md'],'/md')
        if '/md_c' in fin: fout.copy(fin['/md_c'],'/md_c')
        if '/in_' in fin: fout.copy(fin['/in_'],'/in_')
        if '/ref_nstack' in fin: fout.copy(fin['/ref_nstack'],'/ref_nstack') # not correct after rotation ...
        if '/cc_nstack' in fin: fout.copy(fin['/cc_nstack'],'/cc_nstack') # not correct after rotation ...
        add_metadata(fout,'in_rot',in_)
        

        list_couple = fin['md']
        lat         = fin['md/lat'][:]
        lon         = fin['md/lon'][:]
        len_corr    = len(fin['md_c/t'][:])
        for ind,sta in enumerate(fin['md/id'][:]):

            group_ref_id = '/ref/'+sta[0].decode('utf8')+'/'+sta[1].decode('utf8')
            angle    = gps2dist_azimuth(lat[ind,0],lon[ind,0],lat[ind,1],lon[ind,1])    

            if group_ref_id in fout:
                if len(in_['cc_cmp'])==len(fout[group_ref_id]):
                    continue
            data_ref = make_rotation_ref(fin,in_,angle,len_corr,compos,group_ref_id,requested)

            cc_matrix    = False
            if in_['mode'] == 'all' and 'cc' in fin:
                group_cc_id = '/cc/'+ sta[0].decode('utf8') +'/'+sta[1].decode('utf8')
                data_cc     = make_rotation_cc(fin,in_,angle,len_corr,compos,group_cc_id,requested)
                cc_matrix   = True

            for outch in in_['cc_cmp']:
                dset = fout.create_dataset(group_ref_id + '/'+ outch, data = data_ref[:,new_compos.index(outch)],dtype=in_['format'])
                if cc_matrix :
                    dset = fout.create_dataset(group_cc_id + '/'+ outch, data = data_cc[:,:,new_compos.index(outch)],dtype=in_['format'])
            dd.dispc(group_ref_id + " Time : " + str(time.clock() - start),'g','n')

        fout.close()
        fin.close()
        os.remove(lock_file)


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

def make_rotation_ref(fin,in_,angle,len_corr,compos,group,requested):
    data     = np.zeros((len_corr,len(compos)),dtype=in_['format'])
    data_new = np.zeros((len_corr,len(compos)),dtype=in_['format'])
    avl      = np.zeros(np.shape(compos),dtype='int')
    for ch in compos:
        if ch in fin[group]:
            data[:,compos.index(ch)] = fin[group][ch][:].astype(in_['format'])
            if data[:,compos.index(ch)].any():
                avl[compos.index(ch)]    = 1
    #compute rotation
    #data_new = np.array(correction_angle(np.matrix(data),angle[1],0,0,angle[2]+180,0,0),dtype=in_['format'])
    teta1=90.-angle[1]+180.
    teta2=90.-angle[2]+180.
    data_new = np.array(rotation_axes_cart(data,teta1,0,0,teta2,0,0),dtype=in_['format'])
    #test: if all requested input are not available for a specific output => data_new = 0 
    for ich in range(len(compos)):
        if not avl[np.where(requested[ich][:])[0]].all():
            data_new[:,ich] = 0
    return data_new


def make_rotation_cc(fin,in_,angle,len_corr,compos,group,requested):
    nb_date  = len(fin['md_c/date1'])
    data     = np.zeros((nb_date,len_corr,len(compos)),dtype=in_['format'])
    data_new = np.zeros((nb_date,len_corr,len(compos)),dtype=in_['format'])
    teta1=90.-angle[1]+180
    teta2=90.-angle[2]+180
    for ch in compos:
        if ch in fin[group]:
            data[:,:,compos.index(ch)]=fin[group][ch][:].astype(in_['format'])
    for idate in range(nb_date):
        data_cc = data[idate,:,:]
        #compute rotation
        data_new[idate,:,:]=np.array(rotation_axes_cart(np.matrix(data_cc),teta1,0,0,teta2,0,0),dtype=in_['format'])
        #test: if all requested input are not available for a specific output => data_new = 0 
        avl     = np.zeros((len(compos),),dtype='int')
        for ch in compos:       
            if data_cc[:,compos.index(ch)].any():
                avl[compos.index(ch)] = 1             
        for ich in range(len(compos)):
            if not avl[np.where(requested[ich][:])[0]].all():
                data_new[idate,:,ich] = 0
    return data_new


def rotation_axes_cart(data,teta1,phi1,psi1,teta2,phi2,psi2):
    # data     = (time,[XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ])
    # or data  = (time,[EE,EN,EZ,NE,NN,NZ,ZE,ZN,ZZ])
    # toward
    # data_new = (time,[RR,RT,RZ,TR,TT,TZ,ZR,ZT,ZZ])
    # teta     = rotation around Z (counterclockwise), X(E) toward Y(N)
    # psi      = rotation around X (counterclockwise), Y(N) toward Z (to be checked)
    # phi      = rotation around Y (counterclockwise), Z toward X(E) (to be checked)
    data      = np.matrix(data)
    data_new  = np.matrix(np.zeros(data.shape))
    phi1=phi1*np.pi/180.
    phi2=phi2*np.pi/180.
    teta1=teta1*np.pi/180.
    teta2=teta2*np.pi/180.
    psi1=psi1*np.pi/180.
    psi2=psi2*np.pi/180.
    Q_z1=np.matrix([[np.cos(teta1),np.sin(teta1),0],[-np.sin(teta1),np.cos(teta1),0],[0,0,1]])
    Q_z2=np.matrix([[np.cos(teta2),np.sin(teta2),0],[-np.sin(teta2),np.cos(teta2),0],[0,0,1]])
    Q_x1=np.matrix([[1,0,0],[0,np.cos(psi1),np.sin(psi1)],[0,-np.sin(psi1),np.cos(psi1)]])
    Q_x2=np.matrix([[1,0,0],[0,np.cos(psi2),np.sin(psi2)],[0,-np.sin(psi2),np.cos(psi2)]])
    Q_y1=np.matrix([[np.cos(phi1),0,-np.sin(phi1)],[0,1,0],[np.sin(phi1),0,np.cos(phi1)]])
    Q_y2=np.matrix([[np.cos(phi2),0,-np.sin(phi2)],[0,1,0],[np.sin(phi2),0,np.cos(phi2)]])
    Q1=Q_x1*Q_y1*Q_z1
    Q2=Q_x2*Q_y2*Q_z2
    voies=range(0,9)
    ligne1=np.array([0,0,0,1,1,1,2,2,2])
    ligne2=np.array([0,1,2,0,1,2,0,1,2])
    # data_new = Q1 . data . Q2^t
    # data_new_{ij} = rot1_{im}*rot2_{jn}*data_{mn}
    for mm in voies:
        for nn1 in range(0,3):
            for nn2 in range(0,3):
                data_new[:,mm]=data_new[:,mm]+Q1[ligne1[voies[mm]],nn1]*Q2[ligne2[voies[mm]],nn2]*data[:,nn2+nn1*3]
    return data_new


def create_lock_file(filename) : 
    ff=open(filename,'wb')
    pickle.dump(filename,ff)
    ff.close()


def add_metadata(fout,group_name,inu) : 
    ''' add metadata to the output (=processed) hdf5 file which describe the processing applied 
    it contains the name of the function and the input parameters ''' 
    if group_name in fout:
        dd.dispc('WARNING : ' + group_name + ' already esists : no update in /_metadata','r','b')
    else:
        group=fout.create_group(group_name)
        for ikey in inu : 
            try : 
                dset= np.asarray(inu[ikey])
                group.create_dataset(ikey,data=dset)
            except : 
                group.create_dataset(ikey,data=dset.astype('bytes'))  
    return fout 

def mkdir(out_dir) : 
    if os.path.isdir(out_dir) == False :
        os.makedirs(out_dir)


