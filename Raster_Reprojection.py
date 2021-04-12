###############################
# Created on 15.01.2019
#
# @author: goll_ni
###############################


from osgeo import gdal, gdalconst
import glob
from scipy import ndimage
import numpy as np
import sys, os

#   Specify working directory
global cwdir
# cwdir = "/data/temp/temp_perfDEM/"
cwdir = 'T:\SystemPerformance_Proj\perfDEM\CoSSCs_DEM_N47E012/'

def mosaic_folder(folder: str=None):
    """
    :param folder:
    :return:
    """
    #   in terminal:
    #   gdal_merge.py - o merged.tif - of gtiff *

    if folder[-1] is not "/":
        folder = folder + str("/")
    file_list = glob.glob(folder + "*.tif")
    print("Found files:", file_list)
    file_string = " ".join(file_list)
    command = "gdal_merge.py -o merged.tif -of gtiff " + file_string
    os.system(command)


def reproject_raster(src_filename, match_filename, dst_filename):
    #   Input File
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    src_cell_size_X = src_geotrans[1]
    src_cell_size_Y = -src_geotrans[5]

    #   Matching File
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    match_cell_size_X = match_geotrans[1]
    match_cell_size_Y = -match_geotrans[5]

    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    #   If oversampling is present, perform a LP filter
    if src_cell_size_X < match_cell_size_X or src_cell_size_Y < src_cell_size_Y:
        # print("Oversampling Detected")

        ratio_X = match_cell_size_X / src_cell_size_X
        ratio_Y = match_cell_size_Y / src_cell_size_Y

        oversampling_ = max(ratio_X, ratio_Y)
        # print("Oversampling factor:", oversampling_)

        if oversampling_ >= 2:
            # Perform Filtering
            src_filtered = ndimage.uniform_filter(src.GetRasterBand(1).ReadAsArray(), size=int(oversampling_))

            #   Create a new Dataset but with filtered band instead original src band
            new_dataset = gdal.GetDriverByName('GTiff').Create(cwdir + "test.tiff", src.RasterXSize, src.RasterYSize, 1, gdal.GDT_Float32)
            new_dataset.SetGeoTransform(src_geotrans)
            new_dataset.SetProjection(src_proj)

            new_dataset.GetRasterBand(1).WriteArray(src_filtered)

            print("Uniform filtering performed with size:", oversampling_)

            #   Output
            dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdal.GDT_Float32)
            dst.SetGeoTransform(match_geotrans)
            dst.SetProjection(match_proj)

            #   Do the job
            gdal.ReprojectImage(new_dataset, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)

            #   Flush
            new_dataset = None

            #   Remove filtered Dataset
            os.system("rm " + cwdir + "test.tiff")

        else:
            # print("Oversampling negligible")
            dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdal.GDT_Float32)
            dst.SetGeoTransform(match_geotrans)
            dst.SetProjection(match_proj)

            #   Do the job
            gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)

    else:
        #   Output
        dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdal.GDT_Float32)
        dst.SetGeoTransform(match_geotrans)
        dst.SetProjection(match_proj)

        #   Do the job
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)

    # Flush
    src = None
    dst = None
    return gdal.Open(dst_filename, gdalconst.GA_ReadOnly)


if __name__ == '__main__':

    #   Specify acquisition(s) folder(s)
    acquisition_1 = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140823T170012_20140823T170020/")
    acquisition_2 = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20131108T051822_20131108T051830/")

    data_lst = np.zeros(19, dtype=str)

    data_lst[0] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20131001T050959_20131001T051007/")
    data_lst[1] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20131028T051837_20131028T051845/")
    data_lst[2] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140322T170000_20140322T170008/")
    data_lst[3] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140328T165108_20140328T165116/")
    data_lst[4] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140402T165946_20140402T165954/")
    data_lst[5] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140405T052657_20140405T052705/")
    data_lst[6] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140405T052704_20140405T052712/")
    data_lst[7] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140408T165109_20140408T165117/")
    data_lst[8] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140419T165116_20140419T165124/")
    data_lst[9] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140419T165123_20140419T165131/")
    data_lst[10] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140812T170013_20140812T170021/")
    data_lst[11] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140823T165958_20140823T170006/")
    data_lst[12] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140829T165121_20140829T165129/")
    data_lst[13] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140829T165128_20140829T165136/")
    data_lst[14] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140903T170013_20140903T170021/")
    data_lst[15] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20140909T165122_20140909T165130/")
    data_lst[16] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20151026T170012_20151026T170016/")
    data_lst[17] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20160920T170019_20160920T170023/")
    data_lst[18] = str(cwdir + "Koenigsee/TDM1_SAR__COS_BIST_SM_S_SRA_20161217T170013_20161217T170021/")

    #   Specify to-be-reprojected file
    Lidar_Austria = str(cwdir + "Lidar_Austria/Lidar_Austria.tif")

    # #   Specify reference data for reprojection
    # match_1 = str(acquisition_1 + "Output_12m/coSSC_master.dem.tiff")
    # match_2 = str(acquisition_2 + "Output_12m/coSSC_master.dem.tiff")
    #
    # #   Specify output data name
    # dst_1 = str(acquisition_1 + "Lidar_DEM_Reprojected.tiff")
    # dst_2 = str(acquisition_2 + "Lidar_DEM_Reprojected.tiff")

    #   Reprojection
    # try:
    #     reproject_raster(Lidar_Austria, match_1, dst_1)
    # except:
    #     print("Error in Reprojection", match_1)
    #
    # try:
    #     reproject_raster(Lidar_Austria, match_2, dst_2)
    # except:
    #     print("Error in Reprojection", match_2)

    for i in range(0, len(data_lst)):
        try:
            reproject_raster(Lidar_Austria, str(data_lst[i] + "InputDEM"), str(data_lst[i] + "Lidar_DEM_Reprojected.tiff"))
        except():
            print("Error in Reprojection",  str(data_lst[i] + "Lidar_DEM_Reprojected.tiff"))

    print("Process Finished")
