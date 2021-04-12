import os 
import gdal
import numpy as np
from tifffile import imsave

path = r"C:\Users\massi\Downloads\FiumePo\\"
path2 = r"C:\Users\massi\Downloads\FiumePo2\\"

if not os.path.exists(path2):
    os.makedirs(path2)


dateS2_1 = {
    # "01" : ["14", "19"],# "24"],
    "04": ["19", "24"],
    "05": ["19"],#,"24"],
    "06": [ "23", "28"],
    # "07": [ "28"], #["18","23",
# "08": ["12", "17",  "27"], #"22",
    "09": ["26"], #"16", "21", 
    # "12": ["15"]
    }
c = 0
for kkk in dateS2_1.keys():
    month = kkk
    days = dateS2_1[kkk]
    for day in days: 
        name_vv = r"2018_" + month + "_" + day + "_" + "VV.tif"
        file_VV = os.path.join(path,name_vv )
        dataset = gdal.Open(file_VV, gdal.GA_ReadOnly)
        vv = dataset.ReadAsArray()
        dataset = None
        print(vv.shape)
        if c == 0: 
            mean = np.zeros((vv.shape[0], vv.shape[1]))
        c += 1
        mean += vv
mean2 = mean/c
mean22 = mean2.astype('float32')
im = path2 + r"MEAN.tif"
imsave(im, mean22)
        # name_vh = r"2018_" + month + "_" + day + "_" + "VH.tif"
        # file_VV = os.path.join(path,name_vh )
        # dataset = gdal.Open(file_VV, gdal.GA_ReadOnly)
        # vv = dataset.ReadAsArray()
        # dataset = None

        # name_ndwi = r"2018_" + month + "_" + day + "_" + "NDWI.tif"
        # file_VV = os.path.join(path,name_ndwi )
        # dataset = gdal.Open(file_VV, gdal.GA_ReadOnly)
        # vv = dataset.ReadAsArray()
        # dataset = None
for kkk in dateS2_1.keys():
    month = kkk
    days = dateS2_1[kkk]
    for day in days: 
        name_vv = r"2018_" + month + "_" + day + "_" + "VV.tif"
        file_VV = os.path.join(path,name_vv )
        dataset = gdal.Open(file_VV, gdal.GA_ReadOnly)
        vv = dataset.ReadAsArray()
        dataset = None
        vv_diff = vv - mean2
        vv_diff2 = vv_diff.astype('float32')
        im = path2 + r"2018_" + month + "_" + day + "_" + "VV.tif"
        imsave(im, vv_diff2)