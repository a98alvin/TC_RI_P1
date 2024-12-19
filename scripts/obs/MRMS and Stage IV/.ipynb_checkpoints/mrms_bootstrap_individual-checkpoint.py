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

data_type = "MRMS"
SHIPS = pd.read_csv('/Users/acheung/data/SHIPS/all_SHIPs_data_combined_north_atlantic.csv')
SHIPS = SHIPS.where(pd.to_datetime(SHIPS['Time']) >= dt.datetime(2015,5,6)).dropna()
near_land_cases = SHIPS.where(SHIPS['DTL'] <50).dropna()

# def min_distance(point, lines):
#     return lines.distance(point).min()
# dist_list = []
# for ships_inds in near_land_cases.index:

#     coastline = gpd.read_file('/Users/acheung/data/Shapefiles/tl_2024_us_coastline.shp')
#     coastline = coastline.where(
#         (coastline['NAME'] == 'Gulf') | (coastline['NAME'] == 'Atlantic')).dropna()
    
#     PR_inds = list([271,277,list(np.arange(281,304)),list(np.arange(920,935))])

#     def flatten(test_list):
#         if isinstance(test_list, list):
#             temp = []
#             for ele in test_list:
#                 temp.extend(flatten(ele))
#             return temp
#         else:
#             return [test_list]

#     flattened_pr = flatten(PR_inds)
    
#     coastline_no_pr = coastline.drop(flattened_pr)

#     # coastline = gpd.clip(gpd.read_file('ne_10m_coastline.shp'), us).to_crs('EPSG:3087')
#     coastline_no_pr = coastline_no_pr.to_crs('EPSG:3087')
#     points_df = gpd.GeoDataFrame({
#         'geometry': [
#             Point(360-SHIPS['LON'].loc[ships_inds], SHIPS['LAT'].loc[ships_inds])]}, crs='EPSG:4326')
#     points_df = points_df.to_crs('EPSG:3087') # https://epsg.io/3087

#     points_df['min_dist_to_coast'] = points_df.geometry.apply(min_distance, args=(coastline_no_pr,))
#     print((points_df['min_dist_to_coast']/1000)[0])
#     dist = (points_df['min_dist_to_coast']/1000)[0]
#     dist_list.append(dist)
# distances_to_us_coast = pd.DataFrame(dist_list,index = near_land_cases.index)
# distances_to_us_coast.to_csv('/Users/acheung/data/intermediates/dists_to_us_coast_50.csv')

distances_to_us_coast = pd.read_csv('/Users/acheung/data/intermediates/dists_to_us_coast_50.csv')['0']
distances_to_us_coast.index = near_land_cases.index

near_us_coast_cases = near_land_cases.loc[distances_to_us_coast.where(distances_to_us_coast < 100).dropna().index]
atcf_ids = near_us_coast_cases['Storm_ID'].unique()

interpolated_mrms_list = []
imerg_precip_list = []
lat_lon_arr_list = []
time_strings_list = []
individual_times_list = []

ground_mrms_total_list = []
for atcf_it in range(len(atcf_ids)):
# for atcf_it in [8]:
    current_near_land_case = near_us_coast_cases.where(near_land_cases['Storm_ID'] == atcf_ids[atcf_it]).dropna()
    current_storm_name = str(current_near_land_case['Name'].unique()[0])
    current_storm_ID = str(current_near_land_case['Storm_ID'].unique()[0])

    desired_lat = current_near_land_case['LAT'].mean()
    desired_lon = 360-current_near_land_case['LON'].mean()

    start_str = current_near_land_case['Time'].iloc[0]
    end_str = str(pd.to_datetime(current_near_land_case['Time'].iloc[-1]) + dt.timedelta(hours=6)) # Add 6 hours after best-track ends!
    time_strings_list.append(start_str + ', end = ' + end_str)
    
    times_pd = pd.date_range(start=start_str,end=end_str, freq='1h')
    times_pd = times_pd[0:-1] + dt.timedelta(hours=1) # We offset the MRMS time by one hour because the timestamp is at the end of one hour time window

    filepaths = []
    problematic_times = []
    for i in times_pd: # Download all the desired IMERG Images
        # If IMERG file does not exist, download it

        year = i.strftime('%Y')
        month = i.strftime('%m')
        day =  i.strftime('%d')
        hour = i.strftime('%H')

        if (times_pd[0]).to_pydatetime() >= dt.datetime(2020,10,15): # Switch to dual-pol led to file directory change! Keep this in mind!
            data_file_path = '/Users/acheung/data/MRMS/'+year+'/'+month +'/'+ day+\
                '/MultiSensor_QPE_01H_Pass2_00.00_' + year+month+day+'-'+hour+'0000.grib2'
            url_desired = "https://mtarchive.geol.iastate.edu/"+year+"/"+month+"/"+day+"/mrms/ncep/MultiSensor_QPE_01H_Pass2/"\
                + 'MultiSensor_QPE_01H_Pass2_00.00_' + year+month+day+'-'+hour+'0000.grib2.gz'

        elif (times_pd[0]).to_pydatetime() < dt.datetime(2020,10,15): # Switch to dual-pol led to file directory change! Keep this in mind!
            data_file_path = '/Users/acheung/data/MRMS/'+year+'/'+month +'/'+ day+\
                '/GaugeCorr_QPE_01H_00.00_' + year+month+day+'-'+hour+'0000.grib2'
            url_desired = "https://mtarchive.geol.iastate.edu/"+year+"/"+month+"/"+day+"/mrms/ncep/GaugeCorr_QPE_01H/"\
                + 'GaugeCorr_QPE_01H_00.00_' + year+month+day+'-'+hour+'0000.grib2.gz'

    # If day directory does not exist, make day directory
        if os.path.exists('/Users/acheung/data/MRMS/'+year) == False:
            os.mkdir('/Users/acheung/data/MRMS/'+year)

        if os.path.exists('/Users/acheung/data/MRMS/'+year+'/'+month) == False:
            os.mkdir('/Users/acheung/data/MRMS/'+year+'/'+month)

        if os.path.exists('/Users/acheung/data/MRMS/'+year+'/'+month +'/'+ day) == False:
            os.mkdir('/Users/acheung/data/MRMS/'+year+'/'+month +'/'+ day)

        if os.path.exists(data_file_path) == False:

            os.chdir('/Users/acheung/data/MRMS/'+year+'/'+month +'/'+ day)
            try:
                os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies '
                          + url_desired)
                # wget.download(url_desired)
                gunzip(data_file_path+'.gz')
                # os.system('gzip -d '+data_file_path+'.gz')
            except:
                problematic_times.append(i)
                print('Skipped MRMS time ' + str(i))
                continue
        filepaths.append(data_file_path)
        problematic_times_arr = np.asarray(problematic_times) # More or less, remove any times from imerg that mrms did not have
    # ---------------------------------------------------------------------------------------------------------


    # Use multiprocessing to slice MRMS or Stage IV Data (speeds up process significantly)

    def slicing_file(case_loop,filepaths,desired_lon,desired_lat):
        import numpy as np
        import xarray as xr
        import sys
        sys.path.insert(0, '/Users/acheung/TC_RI_P1/scripts/Useful Functions/')
        from distance import distance_calculator

        ground_based_data = xr.open_dataset(filepaths[case_loop], engine="cfgrib")['unknown']
        x, y = np.float32(np.meshgrid(ground_based_data['longitude'], ground_based_data['latitude']))


        # Find index closest to interpolated best-track center or 2-km radar center
        distance_arr = distance_calculator(x,y,(desired_lon,desired_lat))

        abs_dist_arr = (abs(distance_arr))

        min_dist_ind = np.where(abs_dist_arr == np.nanmin(abs_dist_arr))

        # Slice arrays to within 400 indices of desired center

        sliced_lon_grid = x[min_dist_ind[0][0]-400:min_dist_ind[0][0]+400,min_dist_ind[1][0]-400:min_dist_ind[1][0]+400]

        sliced_lat_grid = y[min_dist_ind[0][0]-400:min_dist_ind[0][0]+400,min_dist_ind[1][0]-400:min_dist_ind[1][0]+400]

        sliced_ground_based_data = ground_based_data[min_dist_ind[0][0]-400:min_dist_ind[0][0]+400,min_dist_ind[1][0]-400:min_dist_ind[1][0]+400]

        sliced_ground_based_data_masked = sliced_ground_based_data.where(sliced_ground_based_data.values >=0)
        return np.asarray([sliced_ground_based_data_masked,sliced_lon_grid,sliced_lat_grid])
    #-------------------------------------------------------------------------------------------------------

    # this helper function is needed because map() can only be used for functions
    # that take a single argument (see http://stackoverflow.com/q/5442910/1461210)
    def splat_slicing(args):
        return slicing_file(*args)

    pool = Pool(32)
    iteration_length = len(filepaths)

    def parallel_slicing(iteration_length):
        result_slicing = pool.map(splat_slicing, ((iteration,filepaths,desired_lon,desired_lat
                                                  ) for iteration in range(iteration_length)))
        # return np.asarray(result_slicing), result_slicing[0]['latitude'],result_slicing[0]['longitude']
        return result_slicing

    # ground_total_precip,sliced_lat_1d,sliced_lon_1d= parallel_slicing(iteration_length)
    result_slicing= parallel_slicing(iteration_length)
    mrms_array = np.asarray(result_slicing)
    ground_total_precip = mrms_array[:,0]
    mrms_y_grid = mrms_array[0,2]
    mrms_x_grid = mrms_array[0,1]
    mrms_x_grid = mrms_x_grid - 360
    
    ground_mrms_total_list.append([ground_total_precip,mrms_x_grid,mrms_y_grid,times_pd])
    
#     #-------------------------------------------------------------------------------------------------------
    
#     # Sum up precipitation (MRMS data is hourly QPE)
#     summed_precip_post = np.sum(ground_total_precip,axis=0)

#      #------------------------Process IMERG data based on MRMS Used-------------------------------------------

    imerg_for_mrms_indiv(start_str,end_str,desired_lon,desired_lat,problematic_times_arr)
    imerg_precip_data_pre = xr.open_dataset('/Users/acheung/data/intermediates/imerg_individual/start = '+start_str+', end = '+end_str+' imerg.nc')['__xarray_dataarray_variable__']

    summed_1h_imerg_xr = xr.DataArray(
        data=((imerg_precip_data_pre[0::2].values + imerg_precip_data_pre[1::2].values) * 0.5),
        dims=["time","lon", "lat"],coords=imerg_precip_data_pre[0::2].coords,
        attrs=dict(
            description="IMERG Precip Rate",
            units="mm")) # NOTE THAT THIS IS IMERG TIME, NOT MRMS TIME! Represents the BEGINNING of 1 h time period

    # imerg_precip_data = xr.open_dataset('/Users/acheung/data/intermediates/start = '+start_str+', end = '+end_str+' imerg.nc')['__xarray_dataarray_variable__']
    
    #-------------------------------------Begin Barnes Analysis----------------------------------
    def Barnes_Point_Value(ind_zero_loop, ind_one_loop,imerg_precip_data, current_ground_total_precip,mrms_x_grid,mrms_y_grid,dataset_smoothed):
        # Perform your desired operation
    
        if dataset_smoothed == 'MRMS':
            euclidian_dist = np.sqrt(((imerg_precip_data['lat'].values[ind_zero_loop][ind_one_loop] - mrms_y_grid)**2) +
                                 ((imerg_precip_data['lon'].values[ind_zero_loop][ind_one_loop] - mrms_x_grid)**2))
        elif dataset_smoothed == 'IMERG':
            euclidian_dist = np.sqrt(((imerg_precip_data['lat'].values[ind_zero_loop][ind_one_loop] - imerg_precip_data['lat'].values)**2) +
                         ((imerg_precip_data['lon'].values[ind_zero_loop][ind_one_loop] - imerg_precip_data['lon'].values)**2))

        interested_euclid = pd.DataFrame(euclidian_dist).where(euclidian_dist < 1)
        c = 0.05 # std setting

        weight_func = np.exp(-(interested_euclid**2)/(2*(c**2)))
        norm_weight_func = weight_func/(weight_func.sum().sum())

        if dataset_smoothed == 'MRMS':
            this_point_value = np.nansum(norm_weight_func * current_ground_total_precip)
        elif dataset_smoothed == 'IMERG':
            this_point_value = np.nansum(norm_weight_func * imerg_precip_data.values)

        return this_point_value

#----------------------------Multiprocessing for Barnes Analysis---------------------------------------------


    # this helper function is needed because map() can only be used for functions
    # that take a single argument (see http://stackoverflow.com/q/5442910/1461210)
    def splat_barnes(args):
        return Barnes_Point_Value(*args)

    pool = Pool(48)

    # a pool of 8 worker processes
    def parallel_barnes(M_barnes, N_barnes,imerg_or_mrms,imerg_precip_data,current_ground_total_precip):
        result_barnes_imerg = pool.map(splat_barnes, ((ind_zero_loop, ind_one_loop,imerg_precip_data, 
                                          current_ground_total_precip,mrms_x_grid,mrms_y_grid,imerg_or_mrms) 
                                         for ind_zero_loop in range(M_barnes) for ind_one_loop in range(N_barnes)))

        return np.array(np.asarray(result_barnes_imerg)).reshape(M_barnes, N_barnes)
    
    imerg_barnes_list = []
    mrms_barnes_list = []
    print('Starting Barnes for Storm ' + str(atcf_it + 1))

    for imerg_time_loop_ind in range(len(summed_1h_imerg_xr)):
        imerg_precip_data = summed_1h_imerg_xr[imerg_time_loop_ind]
        current_ground_total_precip = ground_total_precip[imerg_time_loop_ind]
        M_barnes = imerg_precip_data['lat'].values.shape[0]
        N_barnes = imerg_precip_data['lat'].values.shape[1]
        full_barnes_imerg = parallel_barnes(M_barnes, N_barnes,'IMERG',imerg_precip_data,current_ground_total_precip)
        full_imerg_barnes_arr = np.asarray(full_barnes_imerg)

        full_barnes_mrms = parallel_barnes(M_barnes, N_barnes,'MRMS',imerg_precip_data,current_ground_total_precip)
        full_mrms_barnes_arr = np.asarray(full_barnes_mrms)
#---------------------------Mask Data off coastline------------------------------------------
    
        # Mask Ocean from MRMS and IMERG
        # prepare temporary plot and create mask from rasterized map
        proj = {'projection': cartopy.crs.PlateCarree()}
        fig, ax = plt.subplots(figsize=(imerg_precip_data.shape[0]/100, imerg_precip_data.shape[1]/100), dpi=100, subplot_kw=proj)
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        ax.set_frame_on(False)

        ax.add_feature(cartopy.feature.OCEAN, facecolor='black')


        ax.set_extent((float(imerg_precip_data['lon'].min()),float(imerg_precip_data['lon'].max()),
                       float(imerg_precip_data['lat'].min()),float(imerg_precip_data['lat'].max())))
        fig.canvas.draw()
        mask = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        plt.close(fig)

        mask = np.frombuffer(mask, dtype=np.uint8).reshape(nrows, ncols, 3)
        mask = mask.mean(axis=2)
        mask = np.rot90(np.fliplr(mask),k=2)
        masked_mrms  = np.where(mask>0, full_mrms_barnes_arr, np.nan)
        masked_imerg  = np.where(mask>0, full_barnes_imerg, np.nan)
        # imerg_precip_list.append(masked_imerg)
        # interpolated_mrms_list.append(masked_mrms)

        mrms_barnes_list.append(masked_mrms)
        imerg_barnes_list.append(masked_imerg)

        print(str(imerg_time_loop_ind + 1) + ' of ' + str(len(summed_1h_imerg_xr)) + ' Barnes Completed')

#     #-------------------------------------End Barnes Analysis----------------------------------
    interpolated_mrms_list.append(mrms_barnes_list)
    imerg_precip_list.append(imerg_barnes_list)
    lat_lon_arr_list.append([imerg_precip_data_pre['lat'].values, imerg_precip_data_pre['lon'].values])
    individual_times_list.append(summed_1h_imerg_xr['time'].values) # Time is the beginning of the 1 h window
    print(str(atcf_it + 1) + ' of '+ str(len(atcf_ids)) + ' Storms')
    
with open('/Users/acheung/data/intermediates/mrms_list_indiv.pkl', 'wb') as f:
    pickle.dump(interpolated_mrms_list, f)
    
with open('/Users/acheung/data/intermediates/lat_lon_list_indiv.pkl', 'wb') as f:
    pickle.dump(lat_lon_arr_list, f)
    
with open('/Users/acheung/data/intermediates/imerg_list_indiv.pkl', 'wb') as f:
    pickle.dump(imerg_precip_list, f)
    
with open('/Users/acheung/data/intermediates/time_strings_list_indiv.pkl', 'wb') as f:
    pickle.dump(time_strings_list, f)
    
with open('/Users/acheung/data/intermediates/datetimes_indiv.pkl', 'wb') as f:
    pickle.dump(individual_times_list, f)
    
# with open('/Users/acheung/data/intermediates/ground_mrms_total_list.pkl', 'wb') as f:
#     pickle.dump(ground_mrms_total_list, f)