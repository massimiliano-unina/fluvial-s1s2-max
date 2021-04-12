# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:12:59 2020

@author: massi
"""

import gdal 
import os 
# dic = {0: "2018_12_5_", 1 : "2018_9_16_" , 2 : "2019_1_14_", 3 : "2019_9_11_"}
# path_test = r"C:\Users\massi\Downloads\sentinel1_Salerno\Albufera-20200415T192751Z-001\Albufera\File_Training\Input\VV_VH\\"
# path_res = r"C:\Users\massi\Downloads\sentinel1_Salerno\Albufera-20200415T192751Z-001\Albufera\File_Training\Input\results_Albufera\\"
# data_n = ["26022020.tif", "02022020.tif", "05022018.tif", "06022018.tif", "07022020.tif", "08022020.tif", "12022018.tif", "14022020.tif", "17022018.tif", "18022018.tif", "19022020.tif", "20022020.tif", "24022018.tif"]

# for kkk in range(0,len(data_n)): #(1,6): 
#     data = data_n[kkk]
#     dataset = gdal.Open(os.path.join(path_test, "VV_" + data), gdal.GA_ReadOnly)
#     # dataset = gdal.Open(r"C:\Users\massi\Downloads\test"+str(kkk)+"\VH.tif", gdal.GA_ReadOnly)
#     proj   = dataset.GetProjection()
#     geot = dataset.GetGeoTransform()
#     dataset = None     
#     geot2 = [geot[i] for i in range(6)]
#     geot2[3] = geot2[3] #- 320.0
#     # b1 = gdal.Open(r"C:\Users\massi\Downloads\S1_Results\FiumePo"+ str(kkk) +"_FiumePo_Tri2_mobilenetv21_DenseUNet_New3_VH_4_wpatches128.tif", gdal.GA_Update)
#     b1 = gdal.Open(os.path.join(path_res, "input_VV_" + data), gdal.GA_Update)
#     b1.SetProjection(proj)
#     b1.SetGeoTransform(geot2)
#     b1 = None     
#     # b2 = gdal.Open(r"C:\Users\massi\Downloads\S1_Results\FiumePo"+ str(kkk) +"_FiumePo_Tri2_mobilenetv21_DenseUNet_New3_VV_4_wpatches128.tif", gdal.GA_Update)
#     b2 = gdal.Open(os.path.join(path_res, "input_VH_" + data), gdal.GA_Update)
#     b2.SetProjection(proj)
#     b2.SetGeoTransform(geot2)
#     b2 = None     
#     b3 = gdal.Open(os.path.join(path_res, "output_" + data), gdal.GA_Update)
#     # b3 = gdal.Open(r"C:\Users\massi\Downloads\S1_Results\FiumePo"+ str(kkk) +"_FiumePo_Tri2_mobilenetv21_DenseUNet_New3_target_4_wpatches128.tif", gdal.GA_Update)
#     b3.SetProjection(proj)
#     b3.SetGeoTransform(geot2)
#     b3 = None     
# from scipy.misc import imresize

# N_1 = {"139": 3,"168": 4 }

# for n_1 in N_1: 
#     for k1 in range(N_1[n_1]): 

#         b1 = gdal.Open(r"C:\Users\massi\OneDrive\Documenti\corine_"+ str(n_1) + "_"+str(k1)+".tif", gdal.GA_Update)
#         b = b1.GetRasterBand(1)
#         data2 = b.ReadAsArray()
#         print(gdal.Info(r"C:\Users\massi\OneDrive\Documenti\corine_"+ str(n_1) + "_"+str(k1)+".tif"))
#         proj2   = b1.GetProjection()
#         geot2 = b1.GetGeoTransform()
#         print(proj2)
#         print(geot2)
#         ndvi = gdal.Open(r"D:\08_month_40m\\"+ str(n_1) + "_orbit\TS_" + str(k1) + "\channel_vv_hrl_2015\posting_40m\geo_ndvi.tiff")
#         print(gdal.Info(r"D:\08_month_40m\\"+ str(n_1) + "_orbit\TS_" + str(k1) + "\channel_vv_hrl_2015\posting_40m\geo_ndvi.tiff"))
#         ndvi_app = gdal.Open(r"D:\08_month_40m\\"+ str(n_1) + "_orbit\TS_" + str(k1) + "\channel_vv_hrl_2015\posting_40m\geo_ndvi - Copia.tiff", gdal.GA_Update)
#         ndvib = ndvi.GetRasterBand(1)
#         ndvi2 = ndvib.ReadAsArray()
#         data = imresize(data2, size = ndvi2.shape, interp = 'nearest' )
#         ndvi_app.GetRasterBand(1).WriteArray(data)
#         proj   = ndvi.GetProjection()
#         geot = ndvi.GetGeoTransform()
#         print(proj)
#         print(geot)

#         dataset = None 
#         b1.SetProjection(proj)
#         b1.SetGeoTransform(geot)
#         proj2   = b1.GetProjection()
#         geot2 = b1.GetGeoTransform()
#         print(proj2)
#         print(geot2)

#         b1 = None     

#         print(gdal.Info(r"C:\Users\massi\OneDrive\Documenti\corine_139_0.tif"))


# b1 = gdal.Open(r"C:\Users\massi\OneDrive\Desktop\Incendi Boschivi\E.tif")
# b2 = gdal.Open(r"C:\Users\massi\OneDrive\Desktop\Incendi Boschivi\E2.tif")

# b = b1.GetRasterBand(1)
# b22 = b2.GetRasterBand(1)

# data1 = b.ReadAsArray()
# data2 = b22.ReadAsArray()
# data = data1 + data2
# proj2   = b1.GetProjection()
# geot2 = b1.GetGeoTransform()
# print(proj2)
# print(geot2)
# ndvi = gdal.Open(r"C:\Users\massi\OneDrive\Desktop\Incendi Boschivi\E_merge.tif", gdal.GA_Update)
# ndvi.GetRasterBand(1).WriteArray(data)
# ndvi.GetRasterBand(2).WriteArray(data)
# ndvi.GetRasterBand(3).WriteArray(data)
# ndvi.GetRasterBand(4).WriteArray(data)
# ndvi.SetProjection(proj2)
# ndvi.SetGeoTransform(geot2)
# RF_Parma_2020-03-19


b1 = gdal.Open(r"D:\fiumi unsupervised\Dataset_Unsupervised_2021\B4_Parma_2020-03-19.tif" ,gdal.GA_ReadOnly) #Taro_2020-06-20.tif")
proj2   = b1.GetProjection()
geot2 = b1.GetGeoTransform()

# names = ["2019-07-26", "2019-09-14", "2019-06-01", "2020-04-11"]
b2 = gdal.Open(r"C:\Users\massi\Downloads\segmentation_models-master\images\complete_unsupervised_results_2020\DL_Parma_2020-03-19.tif", gdal.GA_Update)
b2.SetProjection(proj2)
b2.SetGeoTransform(geot2)

# for k in range(4):
#     b2 = gdal.Open(r"C:\Users\massi\OneDrive\Documenti\kmeans_7clusters_"+ names[k] +".tif", gdal.GA_Update)
#     b2.SetProjection(proj2)
#     b2.SetGeoTransform(geot2)
#     b2 = gdal.Open(r"C:\Users\massi\OneDrive\Documenti\kmeans_4clusters_"+ names[k] +".tif", gdal.GA_Update)
#     b2.SetProjection(proj2)
#     b2.SetGeoTransform(geot2)

#     # print(proj2)
#     # print(geot2)
#     ndvi = gdal.Open(r"C:\Users\massi\OneDrive\Documenti\Differential_RF_"+ names[k]  +".tif", gdal.GA_Update)
#     ndvi.SetProjection(proj2)
#     ndvi.SetGeoTransform(geot2)
#     # ndvi = gdal.Open(r"C:\Users\massi\OneDrive\Documenti\BCFBM_"+ str(k+1) +".tif", gdal.GA_Update)
#     # ndvi.SetProjection(proj2)
    # ndvi.SetGeoTransform(geot2)    


   
