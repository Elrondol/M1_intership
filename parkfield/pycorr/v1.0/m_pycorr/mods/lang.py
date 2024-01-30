#def parse_options(ind,inu) :
#    """  ind=parse_options(ind,inu) """
#       for ikey in ind.keys() :
#        if inu.has_key(ikey) :
#            ind[ikey]=inu[ikey]
#    return ind


def parse_options(ind,inu) :
    """  ind=parse_options(ind,inu). if ind contains a dict, then we iteratively compare its content to inu"""
    for ikey in ind.keys() :
        if isinstance(ind[ikey],dict) : 
            if ikey in inu :
                    ind[ikey]=parse_options(ind[ikey],inu[ikey])
        elif ikey in inu :
            ind[ikey]=inu[ikey]
    return ind

def merge_options(default,update) :
    """  same as parse_option but add postentiel field that does not exist in default dictionnary"""
    for ikey in default.keys() :
        if isinstance(default[ikey],dict) :
            if ikey in update :
                default[ikey]=parse_options(default[ikey],update[ikey])
        elif ikey in update :
            default[ikey]=update[ikey]
    for ikey in update.keys() :
        if ikey not in default :
           default[ikey]=update[ikey]
    return default
