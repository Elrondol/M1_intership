import h5py

try : 
    import ipdb 
except :
    pass 


def read_group_as_dict(group) :
    db = {}
    for ikey in group:
        #if group[ikey].value.shape > 0 :
        #    db[ikey] = group[ikey][:]
        #else :
        if isinstance(group[ikey], h5py.Dataset) :
            db[ikey] = group[ikey][()]
        elif isinstance(group[ikey], h5py.Group):
            db[ikey] = read_group_as_dict(group[ikey])
        else:
            ipdb.set_trace()
    return db