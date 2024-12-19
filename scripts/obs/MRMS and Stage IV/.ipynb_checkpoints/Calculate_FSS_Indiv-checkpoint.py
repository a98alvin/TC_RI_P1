import os
import pandas as pd
import datetime as dt
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/acheung/TC_RI_P1/scripts/Useful Functions/')
from distance import distance_calculator
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import multiprocessing
from plot_imerg_for_mrms_Individually import imerg_for_mrms_indiv
import geopandas as gpd
import shapely.geometry as geom
from sh import gunzip
import pickle
from multiprocessing import Pool
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore") 


with open('/Users/acheung/data/intermediates/mrms_list_indiv.pkl', 'rb') as file:
    # Load the data from the file
    interpolated_mrms_list = pickle.load(file)
    
with open('/Users/acheung/data/intermediates/lat_lon_list_indiv.pkl', 'rb') as file:
    # Load the data from the file
    lat_lon_arr_list = pickle.load(file)
    
with open('/Users/acheung/data/intermediates/imerg_list_indiv.pkl', 'rb') as file:
    # Load the data from the file
    imerg_precip_list = pickle.load(file)
    
with open('/Users/acheung/data/intermediates/time_strings_list_indiv.pkl', 'rb') as file:
    # Load the data from the file
    time_strings_list = pickle.load(file)
    
with open('/Users/acheung/data/intermediates/datetimes_indiv.pkl', 'rb') as file:
    # Load the data from the file
    individual_times_list = pickle.load(file)
# with open('/Users/acheung/data/intermediates/ground_mrms_total_list.pkl', 'rb') as file:
#     # Load the data from the file
#     ground_mrms_total_list = pickle.load(file)

def fraction_calculator(lat_lon_now,masked_imerg,masked_mrms,imerg_ind_0,imerg_ind_1,radius_to_include,thresholds):
    import sys
    sys.path.insert(0, '/Users/acheung/TC_RI_P1/scripts/Useful Functions/')
    from distance import distance_calculator
    import numpy as np
    import warnings
    import xarray as xr
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # If the thresholds calculate from the same dataset (e.g., IMERG), it means that we are including bias
    imerg_threshold = np.nanpercentile(masked_imerg,thresholds[0])
    mrms_threshold = np.nanpercentile(masked_imerg,thresholds[-1])

    distance_arr_imerg = distance_calculator(lat_lon_now[1],lat_lon_now[0],
                                       (lat_lon_now[1][imerg_ind_0,imerg_ind_1],
                                        lat_lon_now[0][imerg_ind_0,imerg_ind_1]))

    masked_imerg_xr = xr.DataArray(masked_imerg)
    inside_rad_imerg = (masked_imerg_xr.where(distance_arr_imerg < radius_to_include)).values.ravel()
    inside_rad_imerg_no_nan = inside_rad_imerg[~np.isnan(inside_rad_imerg)]
    fraction_rad_imerg = (inside_rad_imerg_no_nan > imerg_threshold).sum()/len(inside_rad_imerg_no_nan)
    
    distance_arr_mrms = distance_calculator(lat_lon_now[1],lat_lon_now[0],
                                       (lat_lon_now[1][imerg_ind_0,imerg_ind_1],
                                        lat_lon_now[0][imerg_ind_0,imerg_ind_1]))
    
    masked_mrms_xr = xr.DataArray(masked_mrms)
    inside_rad_mrms = (masked_mrms_xr.where(distance_arr_mrms < radius_to_include)).values.ravel()
    inside_rad_mrms_no_nan = inside_rad_mrms[~np.isnan(inside_rad_mrms)]
    fraction_rad_mrms = (inside_rad_mrms_no_nan > mrms_threshold).sum()/len(inside_rad_mrms_no_nan)
    return np.asarray([fraction_rad_imerg,fraction_rad_mrms])

from multiprocessing import Pool

# Stack overflow: https://stackoverflow.com/questions/29857498/how-to-apply-a-function-to-a-2d-numpy-array-with-multiprocessing

def splat_f(args):
    return fraction_calculator(*args)

# a pool of worker processes
pool = Pool(86)
M = imerg_precip_list[0][0].shape[0] # Using [0][0] is okay since all images are the same shape
N = imerg_precip_list[0][0].shape[1]

def parallel(fraction_now_ind,indiv_image_loop,M, N,radius_to_include,thresholds):
    lat_lon_now = lat_lon_arr_list[fraction_now_ind]
    masked_imerg = imerg_precip_list[fraction_now_ind][indiv_image_loop]              
    masked_mrms = interpolated_mrms_list[fraction_now_ind][indiv_image_loop]

    result = pool.map(splat_f, ((lat_lon_now,masked_imerg,masked_mrms,
                                  imerg_ind_0,imerg_ind_1,radius_to_include,thresholds) for imerg_ind_0 in range(M) for imerg_ind_1 in range(N)))

    return np.array(np.asarray(result)[:,0]).reshape(M, N),np.array(np.asarray(result)[:,1]).reshape(M, N)

test_radii = np.concatenate([np.arange(10,60,5),np.arange(60,800.1,50)])
thresholds_wanted = [75,90,95,99]

# Calculate fractions in parallel

FSS_lists_with_all_thresholds = [] # appends once per threshold
for thresholds_interested in thresholds_wanted:
    thresholds = [thresholds_interested]
    FSS_overall_list = [] # appends once per storm
    for fraction_now_ind in range(len(imerg_precip_list)): # This loops over all storms
        FSS_per_image_list = [] # appends once per image
        for indiv_image_loop in range(len(imerg_precip_list[fraction_now_ind])): # This loops over all images in a storm
            FSS_list = [] # appends once per radius check
            for radius_loop in test_radii:
                imerg_fractions,mrms_fractions = parallel(fraction_now_ind,indiv_image_loop,M,N,radius_loop,thresholds)
                FBS = (np.nansum((mrms_fractions - imerg_fractions)**2))/len(imerg_fractions.ravel())
                FBS_worst = (np.nansum(mrms_fractions**2) + np.nansum(imerg_fractions**2))/len(imerg_fractions.ravel())
                FSS = 1 - (FBS/FBS_worst)
                FSS_list.append(FSS) # This saves all FSS value to one list, regardless of ROI
            FSS_per_image_list.append(FSS_list)
            print(((thresholds_interested),(fraction_now_ind,len(imerg_precip_list)),(indiv_image_loop,len(imerg_precip_list[fraction_now_ind]))))
        FSS_overall_list.append(FSS_per_image_list)
        print(str(fraction_now_ind + 1) + ' of ' + str(len(imerg_precip_list)) + '; Current Threshold: ' + str(thresholds_interested))
    FSS_lists_with_all_thresholds.append(FSS_overall_list)

with open('/Users/acheung/data/intermediates/FSS_list_indiv.pkl', 'wb') as f:
    pickle.dump(FSS_lists_with_all_thresholds, f)