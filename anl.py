import nibabel as nib
import nilearn.image
import numpy as np
import sys
import os
import re
import pandas as pd


## Net = string pointing to .nii, an np.array, or np.memmap)
## template = string point to a directory of templates, list of files or np.array (4-dimensional if multiple tempaltes)
## Works best when template is a directory of images.

## IMPORTANT: CURRENTLY DESIGNED ONLY TO WORK WITH BINARY MASKS. WHOLE BRAIN CONTINUOUS DATA IN THE WORKS


def jointprob_binary(net,template):

    """
    P(O|T) * P(O|M)
    """

    tempnonzero = np.where(np.resize(template,np.size(template))>0)
    netnonzero = np.where(np.resize(net,np.size(net))>0)
    overlap = len(set(tempnonzero[0]).intersection(netnonzero[0]))


    jointprob = (overlap/len(netnonzero[0]))*(overlap/len(tempnonzero[0]))

    return jointprob


def dice(net,template):
    nonzero=np.where(template+net>0)
    tempnonzero = np.where(np.resize(template,np.size(template))>0)
    netnonzero = np.where(np.resize(net,np.size(net))>0)
    overlap = len(set(tempnonzero[0]).intersection(netnonzero[0]))/len(nonzero[0])
    return overlap


def likelihood(net,template):
    nonzero=np.where(template>0)
    tempnonzero = np.where(np.resize(template,np.size(template))>0)
    netnonzero = np.where(np.resize(net,np.size(net))>0)
    like = len(set(tempnonzero[0]).intersection(netnonzero[0]))/len(nonzero[0])
    return like


def greedy_jointprob_binary(net,templates):

    """
    Net and all Templates (in list) must be of same size
    """

    # Used templates get added to base.
    base = np.zeros(np.shape(templates)[:-1])
    # Predefine output lists
    templatelist=[]
    jplist = []
    templateids=np.arange(0,np.shape(templates)[-1])
    # Preset best joint probabaility and stopping condition
    bestjp = 0
    stopcondition = 0
    while stopcondition == 0:
        # Stop condition, when there are no more templates
        if templates == []:
            stopcondition = 1
        else:
            jp = np.zeros(np.shape(templates)[-1])
            for t in np.arange(0,int(np.shape(templates)[-1])):
                jp[t] = jointprob_binary(net,base+templates[:,:,:,t])
            # Update if jointprobability improves by adding templates
            if np.max(jp) > bestjp:
                maxjp = np.max(jp)
                argmaxjp = np.argmax(jp)
                bestjp = maxjp
                base += templates[:,:,:,argmaxjp]
                jplist.append(maxjp)
                templatelist.append(templateids[argmaxjp])
                np.delete(templates,argmaxjp,axis=3)
                np.delete(templateids,argmaxjp)
            else:
                #Stop condition, when there is no increase when adding more templates
                stopcondition = 1
    return templatelist,jplist





def match_best(net, template, templatenames=[], compareBy='jointprob_binary', q=0, includeNoiseMasks=1, threshold='no'):
    dataOut = match_single(net, template, templatenames=templatenames, compareBy=compareBy, q=q, includeNoiseMasks=includeNoiseMasks, threshold=threshold)
    ids = [dataOut.where(dataOut['input']==n).dropna().index[0] for n in np.unique(dataOut['input'])]
    return dataOut.ix[ids]

def match_single(net, template, templatenames=[], compareBy='jointprob_binary', q=0, includeNoiseMasks=1, threshold='no'):


    [template,net, ftype_template,ftype_net,template_dir]=io_processinput(template,net,q,includeNoiseMasks,threshold=threshold)
    templateN = template.shape[3]
    netN = net.shape[3]
    if isinstance(templatenames,str):
        templatenames=io_loadnames(template_dir,templatenames)
        if includeNoiseMasks==1:
            noisedf = ['Noise (wm)','Noise (csf)']
            templatenames += noisedf
    if includeNoiseMasks==1 and isinstance(templatenames,pd.DataFrame):
        if includeNoiseMasks==1:
            noisedf = pd.DataFrame(data={'name': ['Noise (wm)','Noise (csf)'],'assoc': ['',''],'file': ['','']},index={len(templatenames),len(templatenames)+1})
            templatenames = templatenames.append(noisedf)
        templatenames = templatenames['name']
    if len(templatenames)!=templateN and (len(templatenames)+2!=templateN and includeNoiseMasks==1):
        print('Using numbers as template names.')
        templatenames=list(np.arange(0,templateN))
        if includeNoiseMasks==1:
            templatenames[-2]='Noise (WM)'
            templatenames[-1]='Noise (CSF)'
            print('here')

    #If 4D images, flag this for later
    #net=(net-np.min(net))/(np.max(net)-np.min(net))
    #If template is directory, run for loop through nii files
    comparision_measure=np.zeros([templateN,netN])

    for f in range(0,templateN):
        for n in range(0,netN):
            comparision_measure[f,n]=calc_measure(net[:,:,:,n],template[:,:,:,f],compareBy)

    comparision_measure=np.reshape(comparision_measure,netN*templateN,order='F')

    templateid=list(range(0,len(templatenames)))*netN
    templatenames=list(templatenames)*netN
    inputindex = np.array(range(1,(netN)+1)).repeat(templateN)
    print(len(comparision_measure))
    print(len(templatenames))
    print(len(inputindex))
    print(len(templateid))
    dataOut=pd.DataFrame(data={compareBy : comparision_measure, 'network': templatenames, 'input':inputindex, 'templateid': templateid})
    ascendingCM = False
    dataOut.sort_values(['input', compareBy],inplace=True,ascending=[True, ascendingCM])
    dataOut=dataOut.reset_index(drop=True)
    return dataOut


def match_multi(net, template, templatenames=[], includeNoiseMasks=1, method='greedy', threshold='no', q=0):


    [template,net, ftype_template,ftype_net,template_dir]=io_processinput(template,net,q,includeNoiseMasks,threshold=threshold)
    templateN = template.shape[3]
    netN = net.shape[3]

    if isinstance(templatenames,str):
        templatenames=io_loadnames(template_dir,templatenames)
        if includeNoiseMasks==1:
            noisedf = pd.DataFrame(data={'Noise (wm)','Noise (csf)'},index={len(templatenames),len(templatenames)+1})
            templatenames.append(noisedf)
    if len(templatenames)!=templateN:
        print('Using numbers as template names.')
        templatenames=list(np.arange(0,templateN))
        if includeNoiseMasks==1:
            templatenames[-2]='Noise (WM)'
            templatenames[-1]='Noise (CSF)'

    out = []
    for n in np.arange(0,netN):
        out.append(greedy_jointprob_binary(net[:,:,:,n],template))
    return out




def aux_addtemplate(alltemplate,template):
    if alltemplate.shape[0:3]!=template.shape[0:3]:
        raise ValueError("Shapes need to be the same size")
    else:
        origSize = alltemplate.shape
    alltemplate.reshape(alltemplate.shape[0]*alltemplate.shape[1]*alltemplate.shape[2],alltemplate.shape[3])
    if len(template.shape)==4:
        template=np.sum(template,axis=3)
    alltemplate=np.transpose(alltemplate)
    template.reshape(template.shape[0]*template.shape[1]*template.shape[2])
    template=np.transpose(template)
    newtemplate=alltemplate+template
    newtemplate=np.transpose(newtemplate)
    newtemplate=newtemplate.reshape(origSize)
    newtemplate[newtemplate>0]=1
    return newtemplate



def calc_measure(data,template,measure):
    """
    calc_measure is a wrapper function where it becomes easier to add additinal measures
    """
    if measure=='jointprob_binary':
        x=jointprob_binary(data,template)
    if measure=='dice':
        x=dice(data,template)
    if measure=='likelihood':
        x=likelihood(data,template)
    return x


#Keeps a certain percentage of the image
def threshold_image(image,th=1):
    imageshape = image.shape
    image=np.resize(image,np.size(image))
    cutoff=np.sort(image)[::-1]
    cutoff=cutoff[int(np.round(((th/100)*np.size(image))))-1]
    image[image<cutoff]=0
    image=np.resize(image,imageshape)
    return image




def io_checkdata(dat,datname,single=0,q=0):
    # TODO Add a check to make sure noise masks can be found
    ftype='' # Preassign
    # This should be improved
    if str(type(dat)) == "<class 'numpy.ndarray'>" or str(type(dat)) == "<class 'numpy.core.memmap.memmap'>":
        ftype = 'np'
        if q==0:
            print("Input (" + datname + ") is numpy object (ndarray or memmap)")
#        if single == 1 and len(dat.shape)>3:
#            if dat.shape[3]!=1:
#                raise ValueError("Input (" + datname + ") is larger than a 3d array. Only single nii files can be contrasted with template.")
#                sys.exit(1)
    elif type(dat).__name__ == "Nifti1Image":
        ftype = 'nib'
        if q==0:
            print("Input (" + datname + ") is nibabel image")
    elif os.path.isfile(dat)==True:
        ftype = 'file'
        if q==0:
            print("Input (" + datname + ") is a file")
    elif os.path.isdir(dat)==True:
        ftype = 'dir'
        niiFiles = [x for x in os.listdir(dat) if re.match('.*.nii$',x)]
        if len(niiFiles) == 0:
            raise ValueError("No nii files present in specified directory for input '" + datname + "'': " + dat)
            sys.exit(1)
        if single==1 and len(niiFiles)>1:
            raise ValueError("Only one .nii is allowed for input '" + datname + "' but multiple found in directory: " + dat)
            sys.exit(1)
        if q==0:
            print("Input (" + datname + ") is a directory containing " + str(len(niiFiles)) + " .nii files")
    elif isinstance(dat,str)==True:
        raise ValueError("Input (" + datname + ") interpreted as file or directory but cannot be found: " + dat)
        sys.exit(1)
    elif isinstance(dat,str)==False:
        raise ValueError("Input (" + datname + ") must be a (1) '.nii' file (with nibabel), (2.) string to '.nii' file or directory, (3) numpy array")
        sys.exit(1)

    return ftype

def io_loaddata(dat,datname,ftype,single=0,ftypeout='np',q=0):

    if ftype == 'dir' and single == 1:
        niiFiles = [x for x in os.listdir(dat) if re.match('.*.nii$',x)]
        if len(niiFiles)>1:
            raise ValueError('Too many nii files in directory')
        dat=nib.load(dat + niiFiles[0])
        ftype='nib'
    if ftype != "dir":
        if ftype == 'file':
            dat = nib.load(dat)
            ftype='nib'
        if ftype == 'nib' and ftypeout != 'nib':
            dat = dat.get_data()
    if ftypeout == 'np':
        ftype = io_checkdata(dat,datname,single,1)
    return dat,ftype

def io_loadnames(template,templatenames):
    if os.path.isfile(template + templatenames) == True and os.path.isfile(os.getcwd() + '/' + templatenames) == True and (template != os.getcwd() or template != os.getcwd() + '/'):
        print('WARNING: templatenames exist in both current directory and template directory. Taking template directory')
    if os.path.isfile(template + templatenames) == True:
        templatenames = pd.read_json(template + templatenames)
    elif os.path.isfile(os.getcwd() + '/' + templatenames) == True:
        templatenames = pd.read_json(template + templatenames)
    else:
        print('WARNING: cannot find specified tempaltenames.')
    templatenames.sort_index(inplace=True)
    return list(templatenames.name.values)


def resample(net,template):
    net=nilearn.image.resample_img(net,template.get_affine(),template.get_shape()[0:3])
    return net



def io_processinput(template,net,q=0,includeNoiseMasks=0,threshold='no'):

    #Function outputs net as np and template as np or dir and checks the sizes.
    # Check input Net is correct.
    ftype_net=io_checkdata(net,'net',1,q)
    niiFiles=[]
    template_dir=[]
    if isinstance(template,list)==False:
        ftype_template=io_checkdata(template,'template',0,q)
    #Get data into nibabel format and resample net image if needed
    if ftype_net != 'np':
        net,ftype_net = io_loaddata(net,'net',ftype_net,0,'nib',q)
        net_shape=net.get_shape()[0:3]
    else:
        net_shape=net.shape[0:3]
    if ftype_template != 'np' and ftype_template != 'dir':
        template,ftype_template = io_loaddata(template,'template',ftype_template,0,'nib',q)
        template_shape = template.get_shape()[0:3]
    elif ftype_template == 'dir':
        template_dir = template
        niiFiles = [x for x in os.listdir(template_dir) if re.match('.*.nii$',x)]
        niiFiles = sorted(niiFiles)
        template,ftype_template_example = io_loaddata(template_dir + niiFiles[0],'template_example','file',0,'nib')
        template_shape = template.get_shape()[0:3]
    else:
        template_shape = template.shape[0:3]
    #Resample
    if net_shape!=template_shape:
        if ftype_net != 'np' and ftype_template != 'np':
            print('Input image is different size to template. Resampling.')
            net=resample(net,template)
        else:
            raise ValueError("Input shapes are of different sizes. And one is matrix. Resizing only possible with nii files or directories as input")
    #Convert net into np object.
    net,ftype_net=io_loaddata(net,'net',ftype_net,0,'np',q)
    #make sure nothing is to big.
    if len(template.shape)>4 or len(net.shape)>4:
        raise ValueError('Inputs cannot be bigger than 4. Point to a directory of images instead of a 4d image. (Though this will be implemented soon)')
    if len(net.shape)==4:
        if threshold == 'percent':
            for n in range(0,net.shape[3]):
                net[:,:,:,n]=threshold_image(net[:,:,:,n],1) # <----- add that this parameter can be modified
        if threshold == 'binarize':
            net[net<0]=0
            net[net>0]=1
            print('applying binary threshold')
        else:
            print('no threshold applieds')

    else:
        net = np.expand_dims(net,3)

    if ftype_template == 'dir':
        fid=0
        outSize = list(template.get_shape())
        outSize.append(len(niiFiles))
        template=np.zeros(outSize)
        for f in niiFiles:
            template[:,:,:,fid],ftype_template_file=io_loaddata(template_dir + f,f,'file',0,'np',q)
            fid=fid+1
    else:
        template,ftype_template=io_loaddata(template,'template',ftype_template,0,'np',q)

    if len(template.shape)==3:
        template = np.expand_dims(template,3)

    if includeNoiseMasks==1:
        print('--To be fixed: noise masks must be located in current directory')
        white,ftype_white=io_loaddata(os.getcwd() + '/noiseMasks/' + 'white.nii','white','file',0,'np',q=0)
        csf,ftype_csf=io_loaddata(os.getcwd() + '/noiseMasks/' + 'csf.nii','csf','file',0,'np',q=0)
        white = np.expand_dims(white,3)
        csf = np.expand_dims(csf,3)
        template = np.append(template,white,axis=3)
        template = np.append(template,csf,axis=3)




    return template, net, ftype_template, ftype_net, template_dir
