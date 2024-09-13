import sys
sys.path.insert(0, '/Users/acheung/TC_RI_P1/scripts/Useful Functions/')
import pandas as pd
import datetime as dt
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy import ndimage
from distance import distance_calculator
from cylindrical_conversion import interp_to_equal_grid,circle,x_y_grid_to_cylindrical
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from misc_useful_funcs import multiple_conditions_index_finder

# Select storms before downloading IMERG
# IMERG BEGINS HERE (Filter out anything before)
begin_imerg_YEAR = 2000
begin_imerg_DAY_OF_YEAR = 153
imerg_begin_date = dt.datetime(begin_imerg_YEAR, 1, 1) + dt.timedelta(begin_imerg_DAY_OF_YEAR - 1)

# Open both basin's SHIPS files
SHIPS_NATL = pd.read_csv('/Users/acheung/data/SHIPS/all_SHIPs_data_combined_north_atlantic.csv')
SHIPS_NATL = SHIPS_NATL.where(pd.to_datetime(SHIPS_NATL['Time']) >= imerg_begin_date).dropna()
SHIPS_EPAC = pd.read_csv('/Users/acheung/data/SHIPS/all_SHIPs_data_combined_east_pacific.csv')
SHIPS_EPAC = SHIPS_EPAC.where(pd.to_datetime(SHIPS_EPAC['Time']) >= imerg_begin_date).dropna()
# Reindex EPAC/CPAC to be after NATL
SHIPS_EPAC.index = np.arange(len(SHIPS_NATL),len(SHIPS_NATL) + len(SHIPS_EPAC))

SHIPS = pd.concat([SHIPS_NATL,SHIPS_EPAC])
# Select only tropical storms
tropical_inds = SHIPS.where(SHIPS['TYPE'] == 1).dropna().index

# Select storms at least 50 km from land

dtl_inds = SHIPS.where(SHIPS['DTL'] >=50).dropna().index

# Select Storm with at least 10 KJ/cm2 OHC

ohc_inds =  SHIPS.where(SHIPS['COHC'] >=10).dropna().index

# Select Storm with at least 40% Mid-level (700–500 hPa) RH

rhmd_inds =  SHIPS.where(SHIPS['RHMD'] >=40).dropna().index

result = multiple_conditions_index_finder([tropical_inds,dtl_inds,ohc_inds,rhmd_inds])

# Cases that meet all criteria set above
filtered_cases = SHIPS.iloc[result]

# Download IMERG Data
timedeltas = [-60,-30,0,30] # zero is 0 to 30 minutes and thirty is 30 mins to 1 h
# timedeltas = [0] # zero is 0 to 30 minutes and thirty is 30 mins to 1 h

filepaths = []
desired_time_list = []
for i in filtered_cases.index:
    curr_time_filepaths = []
    for times in timedeltas:
        current_dt_obj = dt.datetime.strptime(filtered_cases['Time'][i],'%Y-%m-%d %H:%M:%S')
        time_desired = current_dt_obj + dt.timedelta(minutes=times)
#         print(i,current_dt_obj,time_desired)

        year = time_desired.year
        day_of_year = time_desired.strftime('%j')
        day_stripped = time_desired.strftime('%Y%m%d')
        time_stripped = time_desired.strftime('%H%M%S')
        min_of_day = str(time_desired.hour*60 + time_desired.minute).zfill(4)
        end_time_stripped = (time_desired+dt.timedelta(minutes=29,seconds=59)).strftime('%H%M%S')

        # If IMERG file does not exist, download it
        IMERG_file_path = '/Users/acheung/data/IMERG/'+str(year)+'/'+day_of_year+'/'+\
            '3B-HHR.MS.MRG.3IMERG.'+day_stripped+'-S'+time_stripped+'-E'+end_time_stripped+\
            '.'+min_of_day+'.V07B.HDF5'
        curr_time_filepaths.append(IMERG_file_path)
        # print(IMERG_file_path)
        # print(os.path.exists(IMERG_file_path))

            # If day directory does not exist, make day directory
        if os.path.exists('/Users/acheung/data/IMERG/'+str(year)) == False:
            os.mkdir('/Users/acheung/data/IMERG/'+str(year))

        if os.path.exists('/Users/acheung/data/IMERG/'+str(year)+'/'+day_of_year) == False:
            os.mkdir('/Users/acheung/data/IMERG/'+str(year)+'/'+day_of_year)

        if os.path.exists(IMERG_file_path) == False:
            url_desired = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/"\
                +str(year)+'/'+day_of_year+'/'+'3B-HHR.MS.MRG.3IMERG.'+day_stripped+'-S'+time_stripped+\
                '-E'+end_time_stripped+'.'+min_of_day+'.V07B.HDF5'
            os.chdir('/Users/acheung/data/IMERG/'+str(year)+'/'+day_of_year)

            os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies '
                      + url_desired)
    filepaths.append(curr_time_filepaths)
    desired_time_list.append(current_dt_obj)

all_time_filepaths = pd.DataFrame(filepaths)
all_time_filepaths.index = filtered_cases.index

case_loop_list = []
all_last_slices = []
progress = 1
for case_loop in all_time_filepaths.index.values: # Run all
# for case_loop in [9875,9876,9877,9878]: # Run all

    # Open IMERG
    print('Index: '+str(case_loop)+'; In Progress: '+str(progress) + ' of ' 
          + str(len(all_time_filepaths.index.values)))
    precip_list = []
    for now_it in range(len(all_time_filepaths.loc[case_loop])):
        fn = all_time_filepaths.loc[case_loop][now_it]
        try:
            f = h5py.File(fn, 'r')
        except: # If the file is corrupt, remove and re-download
            os.system('rm ' + fn)
            os.chdir(fn[0:35])
            new_url = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/' + fn[26:]
            os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies '
                      + new_url)
            f = h5py.File(fn, 'r')


        # Work on precip file
        groups = [ x for x in f.keys() ]
        # print(groups)
        gridMembers = [ x for x in f['Grid'] ]
        # print(gridMembers)

        # Get the precipitation, latitude, and longitude variables
        precip = f['Grid/precipitation'][0][:][:]
        precip = np.transpose(precip)
        precip[precip<-999]=np.nan

        theLats = f['Grid/lat'][:]
        theLons = f['Grid/lon'][:]
        x, y = np.float32(np.meshgrid(theLons, theLats))

        precip_list.append(precip)

    # For now, we take a mean. Might need a different filter in the future
    precip_mean = np.mean(np.asarray(precip_list),axis=0)
    
    centering_lat = filtered_cases.loc[case_loop]['LAT']
    centering_lon = filtered_cases.loc[case_loop]['LON'] # Degrees W
    if centering_lon < 180: # Adjust lon from degrees W to -180 to 180
        centering_lon = centering_lon * -1
    elif centering_lon >= 180:
        centering_lon = centering_lon + 360
    
    distance_arr = distance_calculator(x, y,(centering_lon,centering_lat))

    abs_dist_arr = (abs(distance_arr))

    min_dist_ind = np.where(abs_dist_arr == np.nanmin(abs_dist_arr))
    
    # Slice arrays to within 100 indices of interpolated best-track center

    sliced_lon_grid = x[min_dist_ind[0][0]-75:min_dist_ind[0][0]+75,min_dist_ind[1][0]-75:min_dist_ind[1][0]+75]
    
    if (sliced_lon_grid.shape[0] == 0) or (sliced_lon_grid.shape[1] == 0):
        continue
    sliced_lat_grid = y[min_dist_ind[0][0]-75:min_dist_ind[0][0]+75,min_dist_ind[1][0]-75:min_dist_ind[1][0]+75]
    if (sliced_lat_grid.shape[0] == 0) or (sliced_lat_grid.shape[1] == 0):
        continue
    sliced_precip = precip_mean[min_dist_ind[0][0]-75:min_dist_ind[0][0]+75,min_dist_ind[1][0]-75:min_dist_ind[1][0]+75]
    if (sliced_precip.shape[0] == 0) or (sliced_precip.shape[1] == 0):
        continue
    # NEVER FORGET TO MULTIPLY RADIUS ARRAYS BY DX FOR IMERG PLOTS!!!

    dx = 2
    dy = 2

    # Interpolate IMERG data to equal-distance grid

    eq_lon_grid,eq_lat_grid,eq_dist_data = interp_to_equal_grid(sliced_lon_grid,sliced_lat_grid,
                                                            sliced_precip,dx = dx,dy=dy)
    
    
    distance_arr_sliced = distance_calculator(eq_lon_grid, eq_lat_grid,(centering_lon,centering_lat))

    abs_dist_arr_sliced = (abs(distance_arr_sliced))

    min_dist_ind_sliced = np.where(abs_dist_arr_sliced == np.nanmin(abs_dist_arr_sliced))

    pts_above = eq_dist_data.shape[0] - min_dist_ind_sliced[0][0]
    y_range = np.arange(-min_dist_ind_sliced[0][0],pts_above) * dy

    pts_right = eq_dist_data.shape[1] - min_dist_ind_sliced[1][0]
    x_range = np.arange(-min_dist_ind_sliced[1][0],pts_right) * dx

    xv, yv = np.meshgrid(x_range, y_range)

    x_range.shape[0],np.where(x_range==0)[0][0]

    x_zero_loc = np.where(x_range==0)[0][0]

    if x_zero_loc > x_range.shape[0]/2:
        x_offset = int(x_range.shape[0]-x_zero_loc)
    elif x_zero_loc <= x_range.shape[0]/2:
        x_offset = x_zero_loc

    y_zero_loc = np.where(y_range==0)[0][0]

    if y_zero_loc > y_range.shape[0]/2:
        y_offset = int(y_range.shape[0]-y_zero_loc)
    elif y_zero_loc <= y_range.shape[0]/2:
        y_offset = y_zero_loc

    x_even_inds = x_range[x_zero_loc-(x_offset-1):x_zero_loc+x_offset+1]
    y_even_inds = y_range[y_zero_loc-(y_offset-1):y_zero_loc+y_offset+1]

    sliced_eq_data_pd = pd.DataFrame(eq_dist_data,columns=x_range,index=y_range)

    evenly_sliced_pd = sliced_eq_data_pd[x_even_inds].loc[y_even_inds]

    x_evenly_sliced, y_evenly_sliced = np.meshgrid(x_even_inds, y_even_inds)
    
    angle_rotate = 90 - filtered_cases.loc[case_loop]['SHTD'] # Convert to math angle for rotation
    
    img_rotate = ndimage.rotate(evenly_sliced_pd, angle_rotate, reshape=False,cval=np.nan)
    img_rotate_pd = pd.DataFrame(img_rotate,columns=evenly_sliced_pd.columns,index=evenly_sliced_pd.index)

    pd_last_slice = img_rotate_pd[np.arange(-300,300.1,dx)].loc[-300:300]
    x_evenly_last_sliced, y_evenly_last_sliced = np.meshgrid(pd_last_slice.columns, pd_last_slice.index)
    
    # Save rotated and sliced data to list
    all_last_slices.append(pd_last_slice)

#     # Convert to polar grid
#     cylindrical_data, THETA, rad_arr = x_y_grid_to_cylindrical(
#         center_inds=[0,0],radius_min=0,radius_max=500,radius_interval=1, NOP=1000,
#         x_grid=img_rotate_pd.index.values,y_grid=img_rotate_pd.columns.values,
#         data=img_rotate_pd.values)
    
#     rmw_now = 

#     # Interpolate data to normalized RMW spacings
#     normalized_rad_arr = rad_arr/rmw_now

#     polar_interp = RegularGridInterpolator((normalized_rad_arr,THETA), cylindrical_data,bounds_error=False)

#     theta_mesh,desired_rad_mesh = np.meshgrid(THETA,desired_normalized_rad_arr)
#     polar_rad_norm_data = polar_interp((desired_rad_mesh,THETA))            

#     cylindrical_normalized_list.append(polar_rad_norm_data)
        
    case_loop_list.append(case_loop) 
    progress = progress + 1

finalx,finaly = np.meshgrid(pd_last_slice.columns,pd_last_slice.index)

# NEED TO FIX DATAFRAME INDEX HERE!!! 


finalx,finaly = np.meshgrid(pd_last_slice.columns,pd_last_slice.index)

all_last_slices_xr = xr.DataArray(all_last_slices,dims=["case","x", "y"],coords=dict(
        x=(["x", "y"], finalx),
        y=(["x", "y"], finaly),
        case=(case_loop_list)
    ))

all_last_slices_xr.to_netcdf('/Users/acheung/data/Composite_Data/imerg_NATL_EPAC_WPAC_composites.nc')