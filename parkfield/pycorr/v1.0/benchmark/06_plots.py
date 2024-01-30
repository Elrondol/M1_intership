import m_pycorr.live.c1_md as c1_md

h1 = c1_md.c1_md('C1_04_xcorr_cifalps_conti__xcorr_norm/')

      
#h1.plot_station_map()                 

#h1.get_selection(idvs=['*'],idvr=['*'],icmp=0,filter_p=True,p1=5.,p2=20.,ctype='S',bin_distance=True,bin_width=1.,dist_unit='km',time_unit='s')

h1.plot_selection(idvs=['*'],idvr=['*'],icmp=0,filter_p=True,p1=5.,p2=20.,ctype='S',bin_distance=True,bin_width=1.,tmax = 50,dist_unit='km',time_unit='s')

