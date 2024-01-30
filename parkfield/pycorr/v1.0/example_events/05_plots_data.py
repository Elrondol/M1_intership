import m_pycorr.live.data as ld


a = ld.reader(file_data='data_1.0hz/events/glob/2015-10-26T09-09-42.560000Z_mww_7.5.h5',channels='Z',net='G',stations=None)
b = ld.reader(file_data='data_1.0hz/events/glob/2015-10-26T09-09-42.560000Z_mww_7.5.h5',channels='Z',net='II',stations=None)



b.plot(type='section',dist_degree=True,ev_coord=( a[0].stats.ev_coordinates.latitude, a[0].stats.ev_coordinates.longitude))