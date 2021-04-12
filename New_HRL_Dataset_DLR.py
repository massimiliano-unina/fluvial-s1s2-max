# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:02:43 2020

@author: massi
"""
import imageio 
from matplotlib import pyplot as plt
import numpy as np 
from tifffile import imsave 
import random
import os 
from scipy import ndimage, misc

#############
folder = r"D:\German\\"
dir_list = os.listdir(folder)
dir_list.sort()

N_1 = {"139": 3,"168": 4 }

#N_1 = {"168": 4}
print(dir_list)
N = 6
Out = 3
num = 1

ps1 = 128
r1 = 128


ps = 128
r = 128
enlarge = 64


rotation = [0,45,90, 135, 180, 225, 270, 315]
rotation1 = [0,45,90]
rotation2 = [0] #[0, 135]

A = len(rotation)
B = len(rotation1)
C = len(rotation2)


features = ["NDVI.", "geo_localthetainc.","geo_mean_gamma0dB.","geo_mean_rho6.","geo_rhoLT.","geo_tau.", "IMD_Thre.", "TCD_Thre.","WAW_Thre.","NTCD_Thre."]

patches_iniziali = 0 
patches_finali = 0
for n_1 in N_1: 
    for k1 in range(N_1[n_1]): 
        folder_1 = folder + str(n_1) + "_orbit\TS_" + str(k1) + "\\channel_vv_100m\\"
        x_train = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
        
        y_train = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
        
        x_val = np.ndarray(shape=(0, ps, ps, N), dtype='float32')
        
        y_val = np.ndarray(shape=(0, ps,ps, Out), dtype='float32')
        
        x_test = np.ndarray(shape=(0, ps1, ps1, N), dtype='float32')
        y_test = np.ndarray(shape=(0, ps1,ps1, Out), dtype='float32')
        for feature in features: 
            print(feature)
            if feature == "NDVI.":
                file_out = folder_1 + feature + "tif"
            else: 
                file_out = folder_1 + feature + "tiff"
            
#            command = r'C:\Users\massi\Downloads\OTB-6.6.1-Win64\bin\otbcli_BandMathX -il "' + file_inp + '" -out "' + file_out +'" -exp im1b1'
#            print(command)
#            os.system(command) 
            add = imageio.imread(file_out)
            if feature == "NDVI.":
                ndvi = np.asarray(add)
                print(feature)
            elif feature == "geo_localthetainc.":
                localthetainc = np.asarray(add)
                print(feature)
            elif feature == "geo_mean_gamma0dB.":
                gamma_0 = np.asarray(add)
                print(feature)
            elif feature == "geo_mean_rho6.":
                rho_6 = np.asarray(add)
                print(feature)
            elif feature == "geo_rhoLT.":
                rhoLT = np.asarray(add)
                print(feature)
            elif feature == "geo_tau.":
                tau = np.asarray(add)
                print(feature)
            elif feature == "IMD_Thre.":
                hrl_ARTIFICIAL_SURFACES = np.asarray(add)
                print(feature)
            elif feature == "WAW_Thre.":
                hrl_WATER = np.asarray(add)
                print(feature)
            elif feature == "TCD_Thre.":
                hrl_FOREST = np.asarray(add)
                print(feature)
            elif feature == "NTCD_Thre.":
                hrl_NOFOREST = np.asarray(add)
                print(feature)
#        corine_INVALID = corine_4classes == 0
        size_ndvi = rhoLT.shape
#        ndvi = misc.imresize(ndvi, size_ndvi, interp='bilinear', mode=None)
        # hrl_WATER = waw
        # del waw 
        # hrl_ARTIFICIAL_SURFACES =  imd
        # del imd 
        # hrl_FOREST =  tcd
        # del tcd 
        # hrl_NOFOREST =  ntcd
#        x_mix = np.ndarray(shape=(hrl_NOFOREST.shape[0], hrl_NOFOREST.shape[1], 2), dtype='float32')
#        x_mix[:,:,0] = hrl_NOFOREST
#        x_mix[:,:,1] = hrl_ARTIFICIAL_SURFACES
#        mix_LAYERS = hrl_ARTIFICIAL_SURFACES*hrl_NOFOREST
#        num_pix_Layers = np.sum(mix_LAYERS)
#        del mix_LAYERS
#        mix_LAYERS_2 = np.squeeze(x_mix)
#        del x_mix 
#        mix_LAYERS_2 = np.argmax(mix_LAYERS_2, axis = -1)
#        mask_NOFOREST = hrl_NOFOREST == mix_LAYERS_2
#        del mix_LAYERS_2
#        mask_ARTIFICIAL_SURFACES = 1 - mask_NOFOREST
#        hrl_ARTIFICIAL_SURFACES = hrl_ARTIFICIAL_SURFACES*mask_ARTIFICIAL_SURFACES
#        del mask_ARTIFICIAL_SURFACES 
#        hrl_NOFOREST = hrl_NOFOREST*mask_NOFOREST
#        del mask_NOFOREST
#        mix_LAYERS = hrl_ARTIFICIAL_SURFACES*hrl_NOFOREST
#        num_pix_Layers = np.sum(mix_LAYERS)
#        del mix_LAYERS
#        corine_ARTIFICIAL_SURFACES = corine_4classes == 45
#        corine_FOREST = corine_4classes == 130 
#        corine_NOFOREST = corine_4classes == 215 

        mask_rhoLT = (rhoLT == 0 ) + (hrl_WATER)
#        del corine_4classes
        [s1, s2] = mask_rhoLT.shape
                    
        if k1 < 3:
            p2 = []
            print(len(p2))
            for y in range(1,s1-ps+1,r): 
                for x in range(1,s2-ps+1,r):
                    mask_d0 = mask_rhoLT[y:y+ps,x:x+ps]
                    mask_d0_corine_ARTIFICIAL_SURFACES= hrl_ARTIFICIAL_SURFACES[y:y+ps,x:x+ps]
                    mask_d0_corine_FOREST = hrl_FOREST[y:y+ps,x:x+ps]
                    mask_d0_corine_NOFOREST = hrl_NOFOREST[y:y+ps,x:x+ps]
                    [m1,m2] = mask_d0.shape
                    s_0 =  mask_d0.sum()
                    s_123 = [mask_d0_corine_ARTIFICIAL_SURFACES.sum(), mask_d0_corine_FOREST.sum(), mask_d0_corine_NOFOREST.sum() ]
                    materials = np.where(s_123 == np.max(s_123))[0][0]
                    if s_0 == 0:
                        p2.append([y,x,materials])
            p_train = []
            p_val = []
            
            P1 = len(p2)
            p = p2#[p2[s] for s in v]  
            random.shuffle(p)
            p_train,p_val= p[:int(0.9*P1)],p[int(0.9*P1):P1]
            print(len(p_train))
            print(len(p_val))
    
    #        one_by_one = [ p_train[k][2] for k in range(len(p_train))]
    #        AAA = [ 0 for k in range(len(p_train))]
    #        one_by_one_a = [ one_by_one[k] == AAA[k] for k in range(len(p_train))]
    #        counter_a = np.sum(one_by_one_a)*A
    #        BBB = [ 1 for k in range(len(p_train))]
    #        one_by_one_b = [ one_by_one[k] == BBB[k] for k in range(len(p_train))]
    #        counter_b = np.sum(one_by_one_b)*B
    #        CCC = [ 2 for k in range(len(p_train))]
    #        one_by_one_c = [ one_by_one[k] == CCC[k] for k in range(len(p_train))]
    #        counter_c = np.sum(one_by_one_c)*C
    ##        counter_no = len(p_train) - np.sum(one_by_one_a) - np.sum(one_by_one_b)
    #        print("patches totali :" + str(len(p_train)))
    #        patches_iniziali += len(p_train)
    #        print("patches prima = " + str(np.sum(one_by_one_a)) + " e con augmentation di tipo m0 :" + str(counter_a))
    #        print("patches prima = " + str(np.sum(one_by_one_b)) + " e  con augmentation di tipo m1 :" + str(counter_b))
    #        print("patches prima = " + str(np.sum(one_by_one_c)) + " e  con augmentation di tipo m2 :" + str(counter_c))
    #        print("patches dopo augmentation : " + str((counter_a + counter_b + counter_c)))
    #        
    #        del one_by_one_b 
    #        del BBB 
    #        del one_by_one_c
    #        del CCC 
    #        del one_by_one_a 
    #        del AAA 
    #        P = counter_a + counter_b + counter_c
            P = len(p_train)
            patches_finali += P
            x_train_k = np.ndarray(shape=(P, ps, ps, N), dtype='float32')
            y_train_k = np.ndarray(shape=(P, ps, ps, Out), dtype='float32')
    
            n = 0
            for patch in p_train:
                y0, x0 = patch[0], patch[1]
    #            materials0 = patch[2]
    #            if materials0 == 0: 
    #                for kkkk in rotation:
    #                    gamma_0a = np.pad(gamma_0[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    rhoLTa = np.pad(rhoLT[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    taua = np.pad(tau[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    localthetainca = np.pad(localthetainc[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    gamma_0_1 =  ndimage.rotate(gamma_0a,kkkk, reshape = False) 
    #                    rhoLT_1 = ndimage.rotate(rhoLTa, kkkk, reshape = False)
    #                    tau_1 = ndimage.rotate(taua,kkkk, reshape = False)
    #                    localthetainc_1 = ndimage.rotate(localthetainca,kkkk, reshape = False)
    #                    x_train_k[n,:,:,0] = gamma_0_1[enlarge:ps+enlarge,enlarge:ps+enlarge]
    #                    x_train_k[n,:,:,1] = rhoLT_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    x_train_k[n,:,:,2] = tau_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    x_train_k[n,:,:,3] = localthetainc_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #
    #                    corine_ARTIFICIAL_SURFACESa = np.pad(corine_ARTIFICIAL_SURFACES[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    corine_FORESTa = np.pad(corine_FOREST[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    corine_NOFORESTa = np.pad(corine_NOFOREST[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    ##                    corine_INVALIDa = np.pad(corine_INVALID[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    
    #                    corine_ARTIFICIAL_SURFACES_1 = ndimage.rotate(corine_ARTIFICIAL_SURFACESa,kkkk, reshape = False, order = 0 )#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    #                    corine_FOREST_1 = ndimage.rotate(corine_FORESTa,kkkk, reshape = False, order = 0)#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    #                    corine_NOFOREST_1 = ndimage.rotate(corine_NOFORESTa,kkkk, reshape = False, order = 0)#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    ##                    corine_INVALID_1 = ndimage.rotate(corine_INVALIDa,kkkk, reshape = False, order = 0)#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    #
    #                    y_train_k[n,:,:,0] = corine_ARTIFICIAL_SURFACES_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    y_train_k[n,:,:,1] = corine_FOREST_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    y_train_k[n,:,:,2] = corine_NOFOREST_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    ##                    y_train_k[n,:,:,3] = corine_INVALID_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    n = n + 1
    #            elif materials0 == 1: 
    #                for kkkk in rotation1:
    #                    gamma_0a = np.pad(gamma_0[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    rhoLTa = np.pad(rhoLT[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    taua = np.pad(tau[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    localthetainca = np.pad(localthetainc[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    gamma_0_1 =  ndimage.rotate(gamma_0a,kkkk, reshape = False) 
    #                    rhoLT_1 = ndimage.rotate(rhoLTa, kkkk, reshape = False)
    #                    tau_1 = ndimage.rotate(taua,kkkk, reshape = False)
    #                    localthetainc_1 = ndimage.rotate(localthetainca,kkkk, reshape = False)
    #                    x_train_k[n,:,:,0] = gamma_0_1[enlarge:ps+enlarge,enlarge:ps+enlarge]
    #                    x_train_k[n,:,:,1] = rhoLT_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    x_train_k[n,:,:,2] = tau_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    x_train_k[n,:,:,3] = localthetainc_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #
    #                    corine_ARTIFICIAL_SURFACESa = np.pad(corine_ARTIFICIAL_SURFACES[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    corine_FORESTa = np.pad(corine_FOREST[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    corine_NOFORESTa = np.pad(corine_NOFOREST[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    ##                    corine_INVALIDa = np.pad(corine_INVALID[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    
    #                    corine_ARTIFICIAL_SURFACES_1 = ndimage.rotate(corine_ARTIFICIAL_SURFACESa,kkkk, reshape = False, order = 0 )#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    #                    corine_FOREST_1 = ndimage.rotate(corine_FORESTa,kkkk, reshape = False, order = 0)#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    #                    corine_NOFOREST_1 = ndimage.rotate(corine_NOFORESTa,kkkk, reshape = False, order = 0)#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    ##                    corine_INVALID_1 = ndimage.rotate(corine_INVALIDa,kkkk, reshape = False, order = 0)#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    #
    #                    y_train_k[n,:,:,0] = corine_ARTIFICIAL_SURFACES_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    y_train_k[n,:,:,1] = corine_FOREST_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    y_train_k[n,:,:,2] = corine_NOFOREST_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    ##                    y_train_k[n,:,:,3] = corine_INVALID_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    n = n + 1
    #            elif materials0 == 2: 
    ##                print("sono zero")
    #                for kkkk in rotation2:
    #                    gamma_0a = np.pad(gamma_0[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    rhoLTa = np.pad(rhoLT[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    taua = np.pad(tau[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    localthetainca = np.pad(localthetainc[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    gamma_0_1 =  ndimage.rotate(gamma_0a,kkkk, reshape = False) 
    #                    rhoLT_1 = ndimage.rotate(rhoLTa, kkkk, reshape = False)
    #                    tau_1 = ndimage.rotate(taua,kkkk, reshape = False)
    #                    localthetainc_1 = ndimage.rotate(localthetainca,kkkk, reshape = False)
    ##                    print(gamma_0a.shape)
    ##                    print(gamma_0_1[enlarge:ps+enlarge,enlarge:ps+enlarge].shape)
    #                    x_train_k[n,:,:,0] = gamma_0_1[enlarge:ps+enlarge,enlarge:ps+enlarge]
    #                    x_train_k[n,:,:,1] = rhoLT_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    x_train_k[n,:,:,2] = tau_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    x_train_k[n,:,:,3] = localthetainc_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #
    #                    corine_ARTIFICIAL_SURFACESa = np.pad(corine_ARTIFICIAL_SURFACES[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    corine_FORESTa = np.pad(corine_FOREST[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    corine_NOFORESTa = np.pad(corine_NOFOREST[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    ##                    corine_INVALIDa = np.pad(corine_INVALID[y0:y0+ps,x0:x0+ps],((enlarge,enlarge),(enlarge,enlarge)),'reflect') 
    #                    
    #                    corine_ARTIFICIAL_SURFACES_1 = ndimage.rotate(corine_ARTIFICIAL_SURFACESa,kkkk, reshape = False, order = 0 )#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    #                    corine_FOREST_1 = ndimage.rotate(corine_FORESTa,kkkk, reshape = False, order = 0)#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    #                    corine_NOFOREST_1 = ndimage.rotate(corine_NOFORESTa,kkkk, reshape = False, order = 0)#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    ##                    corine_INVALID_1 = ndimage.rotate(corine_INVALIDa,kkkk, reshape = False, order = 0)#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    #
    #                    y_train_k[n,:,:,0] = corine_ARTIFICIAL_SURFACES_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    y_train_k[n,:,:,1] = corine_FOREST_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    y_train_k[n,:,:,2] = corine_NOFOREST_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    ##                    y_train_k[n,:,:,3] = corine_INVALID_1[enlarge:ps+enlarge, enlarge:ps+enlarge]
    #                    n = n + 1
    
    #            else:
    
                x_train_k[n,:,:,0] = gamma_0[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,1] = rhoLT[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,2] = tau[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,3] = localthetainc[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,4] = rho_6[y0:y0+ps,x0:x0+ps]
                x_train_k[n,:,:,5] = ndvi[y0:y0+ps,x0:x0+ps]
                y_train_k[n,:,:,0]= hrl_ARTIFICIAL_SURFACES[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_train_k[n,:,:,1] = hrl_FOREST[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_train_k[n,:,:,2] = hrl_NOFOREST[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    #                y_train_k[n,:,:,3] = corine_INVALID[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                n = n + 1
            x_train = np.concatenate((x_train, x_train_k))
            del x_train_k
            
            y_train = np.concatenate((y_train, y_train_k))
            del y_train_k
        
            x_val_k = np.ndarray(shape=(len(p_val), ps, ps, N), dtype='float32')
            y_val_k = np.ndarray(shape=(len(p_val), ps, ps, Out), dtype='float32')
    
    
            n = 0
            for patch in p_val:
                y0, x0 = patch[0], patch[1]
                
                x_val_k[n,:,:,0] = gamma_0[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,1] = rhoLT[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,2] = tau[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,3] = localthetainc[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,4] = rho_6[y0:y0+ps,x0:x0+ps]
                x_val_k[n,:,:,5] = ndvi[y0:y0+ps,x0:x0+ps]
                
                y_val_k[n,:,:,0] = hrl_ARTIFICIAL_SURFACES[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_val_k[n,:,:,1] = hrl_FOREST[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_val_k[n,:,:,2] = hrl_NOFOREST[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    #            y_val_k[n,:,:,3] = corine_INVALID[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    
                n = n + 1
            x_val = np.concatenate((x_val, x_val_k))
            del x_val_k
    
            y_val = np.concatenate((y_val, y_val_k))
            del y_val_k
            num +=1
            # np.savez("train_data_HRL_RHO6_NDVI_aug_stride32_with_test_100_" + str(n_1) + "_orbit_TS_" + str(k1) + ".npz", x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val)    
            del x_val
            del y_val
            del x_train
            del y_train
        else: 
            p2 = []
            print(len(p2))
            for y in range(1,s1-ps1+1,r1): 
                for x in range(1,s2-ps1+1,r1):
                    mask_d0 = mask_rhoLT[y:y+ps1,x:x+ps1]
                    mask_d0_corine_ARTIFICIAL_SURFACES= hrl_ARTIFICIAL_SURFACES[y:y+ps1,x:x+ps1]
                    mask_d0_corine_FOREST = hrl_FOREST[y:y+ps1,x:x+ps1]
                    mask_d0_corine_NOFOREST = hrl_NOFOREST[y:y+ps1,x:x+ps1]
                    [m1,m2] = mask_d0.shape
                    s_0 =  mask_d0.sum()
                    s_123 = [mask_d0_corine_ARTIFICIAL_SURFACES.sum(), mask_d0_corine_FOREST.sum(), mask_d0_corine_NOFOREST.sum() ]
                    materials = np.where(s_123 == np.max(s_123))[0][0]
                    if s_0 == 0:
                        p2.append([y,x,materials])
            p_test = p2
            
            P1 = len(p_test)
#            p = p2#[p2[s] for s in v]  
#            random.shuffle(p)
            P = len(p_test)
            patches_finali += P
            x_test_k = np.ndarray(shape=(P, ps1, ps1, N), dtype='float32')
            y_test_k = np.ndarray(shape=(P, ps1, ps1, Out), dtype='float32')
    
            n = 0
            for patch in p_test:
                y0, x0 = patch[0], patch[1]
                x_test_k[n,:,:,0] = gamma_0[y0:y0+ps,x0:x0+ps]
                x_test_k[n,:,:,1] = rhoLT[y0:y0+ps,x0:x0+ps]
                x_test_k[n,:,:,2] = tau[y0:y0+ps,x0:x0+ps]
                x_test_k[n,:,:,3] = localthetainc[y0:y0+ps,x0:x0+ps]
                x_test_k[n,:,:,4] = rho_6[y0:y0+ps,x0:x0+ps]
                x_test_k[n,:,:,5] = ndvi[y0:y0+ps,x0:x0+ps]
                
                y_test_k[n,:,:,0]= hrl_ARTIFICIAL_SURFACES[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_test_k[n,:,:,1] = hrl_FOREST[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                y_test_k[n,:,:,2] = hrl_NOFOREST[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
    #                y_train_k[n,:,:,3] = corine_INVALID[y0:y0+ps, x0:x0+ps]#-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
                n = n + 1
            x_test = np.concatenate((x_test, x_test_k))
            del x_test_k
            
            y_test = np.concatenate((y_test, y_test_k))
            del y_test_k
        
            num +=1

            # np.savez("test_data_HRL_RHO6_NDVI_aug_stride32_with_test_100_" + str(n_1) + "_orbit_TS_" + str(k1) + ".npz",x_test = x_test, y_test= y_test)    
            del x_test
            del y_test


size = 128
size_t = 144 #, 128,128]# 128
n_epochs = 5
n_batch = 16
l_rate = 0.001
#"case_A",
comb = ["case_3_bNDVI"]#["case_11"]# ["case_10"]# [ "case_3_bis", "case_6_bis", "case_1", "case_2", "case_3_wo_Tau", "case_7_Andrew", "case_3",  "case_4", "case_5","case_7",  "case_8", "case_9"] #  ["case_5_HRL"] #
num =  [3, 5, 2, 3, 2, 3, 4, 5, 6, 3, 4, 5]#[4]#[3]# # [6] #  #
N = np.max(num)#[0]
dic = {} 
for kk in range(len(comb)):
    dic[comb[kk]] = num[kk]
n_classes = 3    

N_1 = {"139": 3,"168": 4 }
x_train1 = np.ndarray((0, size, size, N), dtype='float32')
y_train1 = np.ndarray((0, size, size, n_classes), dtype='float32')
x_val1 = np.ndarray((0, size, size, N), dtype='float32')
y_val1 = np.ndarray((0, size, size, n_classes), dtype='float32')
x_test1 = np.ndarray((0, size, size, N), dtype='float32')
y_test1 = np.ndarray((0, size, size, n_classes), dtype='float32')

for n_1 in N_1: 
   for k1 in range(N_1[n_1]): 
       print(str(n_1) + "_orbit_TS_" + str(k1))
       if k1 == 3:
           print("test" + str(n_1) + "_orbit_TS_" + str(k1))
           train_val = np.load("test_data_HRL_RHO6_NDVI_aug_stride32_with_test_100_" + str(n_1) + "_orbit_TS_" + str(k1) + ".npz")#,x_test = x_test, y_test= y_test, x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val)    
           x_test1 = train_val['x_test']
           y_test1 = train_val['y_test']
#            x_test1 = np.concatenate((x_test1, x_test_2[:, :size,:size,:]))
           print(x_test1.shape)
#            y_test1 = np.concatenate((y_test1, y_test_2[:, :size,:size,:]))
           print(y_test1.shape)
       else: 
           print("train" +  str(n_1) + "_orbit_TS_" + str(k1))
           train_val = np.load("train_data_HRL_RHO6_NDVI_aug_stride32_with_test_100_" + str(n_1) + "_orbit_TS_" + str(k1) + ".npz")#,x_test = x_test, y_test= y_test, x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val)    
   
           x_train_2 = train_val['x_train']
           
           print(x_train1.shape)
           y_train_2 = train_val['y_train']
           x_val_2 = train_val['x_val']
           y_val_2 = train_val['y_val']
           print(x_train_2.shape)
           x_train1 = np.concatenate((x_train1, x_train_2[:n_batch*int(675/5), :size,:size,:]))
           print(x_train1.shape)
           y_train1 = np.concatenate((y_train1, y_train_2[:n_batch*int(675/5), :size,:size,:]))
           print(y_train1.shape)
           x_val1 = np.concatenate((x_val1, x_val_2[:n_batch*int(675/5), :size,:size,:]))
           print(x_val1.shape)
           y_val1 = np.concatenate((y_val1, y_val_2[:n_batch*int(675/5), :size,:size,:]))
           print(y_val1.shape)
   #        x_test1 = train_val['x_test']
   #        y_test1 = train_val['y_test']
np.savez("train_HRL.npz",x_test = x_test1, y_test= y_test1, x_train = x_train1, y_train = y_train1, x_val = x_val1, y_val = y_val1)    
