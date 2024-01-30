def lst(input) :
    nel=len(input)
    dispc_help('this list has '+str(nel)+' elements','','',4)

    k=0
    for iel in input :
        dispc_help(str(k),'',input[k],4)
        k=k+1 
        if k> 10 :
            dispc('...','b','b')
            dispc_help(str(nel),'',input[-1],4)
            return 




def dd(input,tab=0) :
    """ print keys & data  of object input with fancy colors """
    print(type(input))
#    if str(type(input))=="<type 'instance'>" : 
#        dict_=input.__dict__
    if type(input)==dict :
        dict_=input
#    if type(input)==list :
#        dispc('input is a list : printing the dictionary of the 1st node','g','b')
#        dict_=input[0].__dict__
#    if str(type(input)).split()[0]=='<class':
#        dict_=input.__dict__
#    if type(input)==dict :
#        dict_=input.keys()

    #sert a aligner les : apres les nom des cles. 
    length=0 
    for ikey in dict_ : 
        if len(ikey) > length : 
            length=len(ikey)

    #tab/last tab sert a tabuler les cles des sous-object (qui sont des listes)
    last_tab=tab    
    for ikey in dict_ :
        type_str=str(type(dict_[ikey])).split("'")[1]
        type_key=type(dict_[ikey])
        key_str=ikey.rjust(len(ikey)+last_tab)
        
        if type_str=='list' :
            type_str=type_str+'['+str(len(dict_[ikey]))+']'
        dispc_help(key_str,type_str,'  '+str(dict_[ikey]),length+last_tab)

        continue
        if type_key==list : 
            if len(dict_[ikey]) > 0 :
                last_tab=last_tab+4
                dd(dict_[ikey][0],last_tab)
                last_tab=last_tab-4 

def dispc_twice(text1,text2,length) :
    prefix='\x1b[1m \x1b[7m';
    str_=prefix+'\x1b['+str(36)+'m'+text1.ljust(length,' ')+':'+'\x1b[0m';
    str_=str_+'\x1b[1m \x1b['+str(36)+'m  '+text2.ljust(5,' ')+'\x1b[0m';
    print(str_)


def dispc_help(text1,text2,text3,length) :
    prefix='\x1b[1m';
    str_=prefix+'\x1b['+str(31)+'m'+text1.ljust(length,' ')+':'+'\x1b[0m';
    str_=str_+'\x1b['+str(33)+'m  '+text2.ljust(5,' ')+'\x1b[0m';
    str_=str_+'\x1b['+str(36)+'m'+text3[0:50]+'\x1b[0m';
    print(str_)
        

#------------------------------------------------
def dispc(text,color,attr) :
    # print text in color with attribute attr ! 
    if color =='r' :
        ncolor=31
    elif color=='g' :
        ncolor=32 
    elif color=='y' :
        ncolor=33
    elif color=='b' :
        ncolor=34
    elif color=='m' :
        ncolor=35
    elif color=='c' :
        ncolor=36
    elif color=='w' :
        ncolor=37
    elif color=='gray' :
        ncolor=38

    if attr=='b' :
        prefix='\x1b[1m';
    elif attr=='d' :
        prefix='\x1b[2m';
    elif attr=='u' :
        prefix ='\x1b[4m';
    elif attr=='blink' :
        prefix ='\x1b[5m'
    elif attr=='r' :
        prefix='\x1b[7m'
    else :
        prefix=''


    str_=prefix+'\x1b['+str(ncolor)+'m'+text+'\x1b[0m';
    print(str_)
    
    
    
    
    

