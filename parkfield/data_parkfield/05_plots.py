import m_pycorr.live.c1_md as c1_md

h1 = c1_md.c1_md('C1_04_xcorr_glob_conti__xcorr_norm/')

      
#h1.plot_station_map()
h1.plot_selection(idvs=['*'],idvr=['*'],icmp=0,filter_p=True,p1=10.,p2=100.,ctype='S',
    bin_distance=True,bin_width=1.,dist_unit='deg',time_unit='s',norm=1,norm_tr=False,
    save_plot=True,file_name='./noise_corr.png')
