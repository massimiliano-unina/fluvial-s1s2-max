# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:41:30 2020

@author: Antonio
"""

import os
import imageio
import numpy as np
import gdal
import geopandas as gpd
from shapely.geometry import Polygon, mapping

def save_img(img,folder,geo,proj):
    
    imageio.imwrite(folder,img)
    hh = gdal.Open(folder,gdal.GA_Update)
    hh.SetGeoTransform(geo)
    hh.SetProjection(proj)
    
    return

def open_image(image_path):

    image = gdal.Open(image_path)
    band = image.GetRasterBand(1)
    data_type = band.DataType
    cols = image.RasterXSize
    rows = image.RasterYSize
    geotransform = image.GetGeoTransform()
    proj = image.GetProjection()
    minx = geotransform[0]
    maxy = geotransform[3]
    maxx = minx + geotransform[1] * cols
    miny = maxy + geotransform[5] * rows
    X_Y_raster_size = [cols, rows]
    extent = [minx, miny, maxx, maxy]
    information = {}
    information['geotransform'] = geotransform
    information['extent'] = extent
    information['X_Y_raster_size'] = X_Y_raster_size
    information['projection'] = proj
    information['data_type'] = data_type
    image_array = np.array(image.ReadAsArray(0, 0, cols, rows))
    return image_array, information

def save_image (image_to_save, path_to_save, driver_name, datatype, geotransform, proj, NoDataValue = None):
    # adfGeoTransform[0] /  top left x  /
    # adfGeoTransform[1] /  w - e pixel resolution  /
    # adfGeoTransform[2] /  rotation, 0 if image is "north up"  /
    # adfGeoTransform[3] /  top left y  /
    # adfGeoTransform[4] /  rotation, 0 if image is "north up"  /
    # adfGeoTransform[5] /  n - s pixel resolution  /

    import gdal
    import numpy as np
    driver = gdal.GetDriverByName(driver_name)

    if len(np.shape(image_to_save)) == 2:
        bands = 1
        cols = np.shape(image_to_save)[1]
        rows = np.shape(image_to_save)[0]

    if len(np.shape(image_to_save)) > 2:
        bands = np.shape(image_to_save)[2]
        cols = np.shape(image_to_save)[1]
        rows = np.shape(image_to_save)[0]

    outDataset = driver.Create(path_to_save, cols, rows, bands, datatype)

    outDataset.SetGeoTransform(geotransform)
    if proj != None:
        outDataset.SetProjection(proj)

    # Set no data values
    if outDataset.RasterCount > 1:
        for i in range(1, outDataset.RasterCount):
            outDataset.GetRasterBand(i).SetNoDataValue(NoDataValue)
    else:
        outDataset.GetRasterBand(1).SetNoDataValue(NoDataValue)

    if bands > 1:

        for i in range(1, bands + 1):
            outDataset.GetRasterBand(i).WriteArray(image_to_save[:, :, (i - 1)], 0, 0)

    else:
        outDataset.GetRasterBand(1).WriteArray(image_to_save, 0, 0)

    outDataset = None
    return;
    

folder = r"D:\Earthlitycs\Perdite_Idriche"
# points = "D:\\Earthlitycs\\punti_campionati\\punti_campionati.shp"

zone = os.listdir(folder)
zone.sort()


window_size = 35


for zona in zone:
    
    images = os.listdir(os.path.join(folder,zona))
    
    for img in images:
        
        if img[-6:-4] == 'ps':
            
            ii = gdal.Open(os.path.join(folder,zona,img))
            geo = ii.GetGeoTransform()
            proj = ii.GetProjection()
            
            
            ms = imageio.imread(os.path.join(folder,zona,img))
            
            b8 = ms[:,:,7]
            b7 = ms[:,:,6]
            b5 = ms[:,:,4] 
            b1 = ms[:,:,0]
            
            ndvi = (b8-b5)/(b8+b5)
            mndvi = (b7-b5)/(b7+b5-(2*b1))
            osavi = 1.16*(b7-b5)/(b7+b5+0.16)
            savi = 1.5*(b7-b5)/(b7+b5+0.5)
            
            if not os.path.exists(os.path.join(folder,zona,'Indices')):
                os.makedirs(os.path.join(folder,zona,'Indices'))
            
            save_img(ndvi, os.path.join(folder,zona,'Indices','NDVI_' + img), geo, proj)
            save_img(mndvi, os.path.join(folder,zona,'Indices','MNDVI_' + img), geo, proj)
            save_img(savi, os.path.join(folder,zona,'Indices','SAVI_' + img), geo, proj)
            save_img(osavi, os.path.join(folder,zona,'Indices','OSAVI_' + img), geo, proj)
            
            # imageio.imwrite(os.path.join(folder,zona,'Indices','NDVI_' + img),ndvi)
            # imageio.imwrite(os.path.join(folder,zona,'Indices','MNDVI_' + img),mndvi)
            # imageio.imwrite(os.path.join(folder,zona,'Indices','OSAVI_' + img),osavi)
            # imageio.imwrite(os.path.join(folder,zona,'Indices','SAVI_' + img),savi)
            
            # hh = gdal.open(os.path.join(folder,zona,'Indices','NDVI_' + img),gdal.GA_Update)
            # hh.SetGeoTransform(geo)
            # hh.SetProjection(proj)
             # Shapefile of points for calibration
            
            points = os.path.join(folder,zona,'punto.shp')
            
            points2 = gpd.read_file(points)
            crs_shp = points2.crs
            
            ms1, ms1_info = open_image(os.path.join(folder,zona,img))
            # Reference_DEM_invalid_pixels_mask = Reference_DEM == no_data_value
        
            ulX = ms1_info['geotransform'][0]
            ulY = ms1_info['geotransform'][3]
            xRes = ms1_info['geotransform'][1]
            yRes = ms1_info['geotransform'][5]
            
            
               # I iterate the points of shapefile and I extract the relative values of these positions on rasters
            
            point_dic= {}
            
            proji = ms1_info['projection']
            # pints1 = ponts2
            points1 = points2.to_crs(proji)
            for index, row in points1.iterrows():
                point_id = row.fid
                point_dic['fid']=[]
                point_dic['fid'].append(point_id)
                # print(row.id)
                coordinates_of_point = mapping(row.geometry)['coordinates']  # (East, North)
                E = coordinates_of_point[0]
                N = coordinates_of_point[1]
                # I pass from geographic coordinates to image coordinates
                x = int((coordinates_of_point[0] - ulX) / xRes)
                y = int((coordinates_of_point[1] - ulY) / yRes)
                # I check if there are invalid pixels in the area
                ndvi1 = ndvi[x - int(window_size / 2): x + int(window_size / 2),
                                                y - int(window_size / 2): y + int(window_size / 2)]
                
                geotransform2 = []
                xx = xRes*(x - int(window_size / 2)) + ulX
                yy = yRes*(y - int(window_size / 2)) + ulY
                geotransform2.append(xx)
                geotransform2.append(ulX)
                geotransform2.append(0)
                geotransform2.append(yy)
                geotransform2.append(0)
                geotransform2.append(ulY)
                print(ndvi1)
                print(zona)
                save_image(ndvi1, os.path.join(folder,zona,'Indices','NDVI_crop_' + str(window_size) + '.tiff'), "GTiff", ms1_info['data_type'], geotransform2, proji,-9999)