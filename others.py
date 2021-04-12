# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:26:40 2017

@author: serg.vitale
"""
import sys
#import scipy.io as sio
import os
import numpy as np
from scipy import ndimage
import scipy.special as sspe
#import math
import xml.etree.ElementTree as ET

def get_attribute(children,node):
            name=children.get('name')
            t=children.get('type')            
            if t=='int':
                node[name]=int(children.text)
            elif t=='double':
                node[name]=float(children.text)
            elif t=='bool':
                node[name]=bool(int(children.text))
            else:
                node[name]=children.text
            
def parser_xml(filename):
    tree=ET.parse(filename)
    root=tree.getroot()
    model={}
    for child in root:
        name1=child.get('name')
        node={}
        for children in child:
            if len(children)==0:
                get_attribute(children,node)
            else:
                name2=children.get('name')
                node[name2]={}                
                for childs in children:
                    get_attribute(childs,node[name2])
        model[name1]=node
                     
    return model
        
def gaussian2d (N, std):
    
    t=np.arange(-(N-1)/2,(N+2)/2)
    t1,t2=np.meshgrid(t,t)
    std=np.double(std)
    w = np.exp(-0.5*(t1/std)**2)*np.exp(-0.5*(t2/std)**2) 
    return w
    
def kaiser2d (N, beta):
    
    t=np.arange(-(N-1)/2,(N+2)/2)/np.double(N-1)
    t1,t2=np.meshgrid(t,t)
    t12=np.sqrt(t1*t1+t2*t2)
    w1=np.kaiser(N,beta)
    w=np.interp(t12,t,w1)
    w[t12>t[-1]]=0
    w[t12<t[0]]=0
    
    return w
    
#def export(I_F, i=0,output_path=None):#network=None):
#    """
#    save data in a matlab file to calculate performance index
#    
#        I_F:               MS pansharpened(output)
#        img:               number of image on test        
#	output_path:	   desired output folder        
#	"""    
#        
#    result={'I_out':I_F}
#            
#    if not os.path.exists(output_path):
#        os.makedirs(output_path)
#    
#    sio.savemat(output_path+'/output%03d.mat'%(i+1),result)
#    
#def export2(filtered,speckle,output_path,i=0):#network=None):
#    
#    #intensity image    
#    result={}
#    result['I_out']=filtered
#    result['I_speckle']=speckle
#    
#    if not os.path.exists(output_path):
#        os.makedirs(output_path)
#    sio.savemat(output_path+'/output%03d.mat'%(i+1),result)
#
#def export3(filtered,speckle,output_path,img):#network=None):
#    
#    #intensity image    
#    result={}
#    result['I_out']=filtered
#    result['I_speckle']=speckle
#    
#    if not os.path.exists(output_path):
#        os.makedirs(output_path)
#    sio.savemat(output_path+'/%s.mat'%(img),result)

#def saveModel(layer, model,v_loss, residual):
#    model=saveLayer(layer, model)
#    
#    vloss={'v_loss':v_loss}
#    
#    filt1=model['layers'][0].shape
#    path='./networks/{0}_{4}/{1}_{2}_{3}'.format(model['sensor'],
#          filt1[1],filt1[2],filt1[3],model['block_size'])
#    if not os.path.exists(path):
#        os.makedirs(path)
#        
#    if residual:
#        name1='/PNN_model_{}_residual.mat'.format(model['epochs'])
#        name2='/v_loss_{}_residual.mat'.format(model['epochs'])
#    else:
#        name1='/PNN_model_{}.mat'.format(model['epochs'])
#        name2='/v_loss_{}.mat'.format(model['epochs'])
#    sio.savemat(path+name1, model)
#    sio.savemat(path+name2, vloss)
    
    
def saveLayer(layer, model):
    """add trained layer to model
        layer: list of layers after training
        model: PNN_model
    """
    padsize=0
    layers=[]
    for l in layer:
        w,b = [np.asarray(l.w.eval()), np.asarray(l.b.eval())]
        layers.append(w)
        layers.append(b)
        padsize+=l.w.eval().shape[2]-1
    model['padSize']=padsize
    model['layers']=layers
    model['ratio']=4
    return model

def saveBN_layer(layer,model):
    # layer: list of layers to save
    # model: model in which save layer

    padsize=0
    layers=[]
    for l in layer:
        if l.BN==False:
            w,b = [np.asarray(l.w.eval()), np.asarray(l.b.eval())]
            layers.append(w)
            layers.append(b)
        else:
            w=np.asarray(l.w.eval())
            gamma=np.asarray(l.BNlayer.gamma.eval())
            beta,mean,var=[np.asarray(l.BNlayer.beta.eval()),
                                np.asarray(l.BNlayer.mean.eval()),
                                np.asarray(l.BNlayer.var.eval())]
            layers.append(w)
            layers.append(gamma)
            layers.append(beta)
            layers.append(mean)
            layers.append(var)
        padsize+=l.w.eval().shape[2]-1
    model['padSize']=padsize
    model['layers']=layers
    model['ratio']=4
    return model
        


def interp23(image, ratio):
    if (2**round(np.log2(ratio)) != ratio):
#        print 'Error: only resize factors of power 2'
        return

    b,r,c = image.shape

    CDF23 = 2*np.array([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0, -0.000060081482])
    d = CDF23[::-1] 
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23
    
    first = 1
    for z in range(1,np.int(np.log2(ratio))+1):
        I1LRU = np.zeros((b, 2**z*r, 2**z*c))
        if first:
            I1LRU[:, 1:I1LRU.shape[1]:2, 1:I1LRU.shape[2]:2]=image
            first = 0
        else:
            I1LRU[:,0:I1LRU.shape[1]:2,0:I1LRU.shape[2]:2]=image
        
        for ii in range(0,b):
            t = I1LRU[ii,:,:]
            for j in range(0,t.shape[0]):
                t[j,:]=ndimage.correlate(t[j,:],BaseCoeff,mode='wrap')
            for k in range(0,t.shape[1]):
                t[:,k]=ndimage.correlate(t[:,k],BaseCoeff,mode='wrap')
            I1LRU[ii,:,:]=t
        image=I1LRU
        
    return image

#def get_loss(path,cost):
#    """ Salva loss distinguendo il caso L2 training con L1 training:
#        Nel caso L2 salvo solo una loss.
#        Nel caso L1 salvo sia loss l1 che l2
#        """
#    if cost=='L2':
#        vloss=sio.loadmat(path,squeeze_me=True)
#        vloss=vloss['v_loss']
#        return vloss
#    elif cost=='L1':
#        vloss=sio.loadmat(path,squeeze_me=True)
#        vloss1=vloss['v_loss1']
#        vloss2=vloss['v_loss2']
#        return vloss1,vloss2
        
#def get_loss2(path):
#    """
#    salva le loss senza distingure i casi delle funzioni di costo a differenza di get_loss()
#    creata perchè nelle reti più recenti salvo due loss anche nel caso di training L2
#    Nel caso di loss L1: vloss1=validation loss L1; vloss2=validation loss L2
#    Nel caso di loss L2:vloss1=vloss2= vladitaion loss l2 
#    """
#    vloss=sio.loadmat(path,squeeze_me=True)
#    vloss1=vloss['v_loss1']
#    if vloss.has_key('v_loss3'):
#        vloss2=vloss['v_loss3']
#    else:
#        vloss2=vloss['v_loss2']
#    return vloss1,vloss2
        
def save_loss(path,loss,pnt,pnt1=None):
    """salva pnt  punti della loss:
        salta la epoca zero, prende epoca 1, poi da epoca 50 ne prende ogni 50
        -pnt1!=0:  salva pnt1 punt per le prime 1000 epoche e (pnt-pnt1) per le restanti 9000        
        -pnt=0:    salva pnt punti della loss
        """
    if pnt1!=None:
        pt1=1000/pnt1
        pt2=9000/(pnt-pnt1)   
        vl=[j for (i,j) in zip(xrange(len(loss)),loss*10000) if ((i<=1000 and i%pt1==0) or(i>1000 and i%pt2==0))]       
        epoch=[i for i in xrange(len(loss)) if ((i<=1000 and i%pt1==0) or(i>1000 and i%pt2==0))]
    else:
        pt1=len(loss)/pnt 
        vl=loss[0:len(loss+1):pt1]
#        vl[0]=loss[1]
#        vl=[j for (i,j) in zip(xrange(len(loss)),loss*10000) if (i==1 or i%pt1==0) ]       
#        epoch=[i for i in xrange(len(loss+1)) if ( i==1 or i%pt1==0)]
        epoch=np.arange(0,len(loss+1),pt1)
#        epoch[0]=1

    np.savetxt(path,zip(epoch,vl),fmt='%.6e')
    
    


    
    
