def imerg_splat_slicing(args):
    return process_imerg(*args)

def process_imerg(case_loop,filepaths_imerg,desired_lon,desired_lat):
    import numpy as np
    import xarray as xr
    import sys
    import os
    import h5py
    sys.path.insert(0, '/Users/acheung/TC_RI_P1/scripts/Useful Functions/')
    from distance import distance_calculator
    fn = filepaths_imerg[case_loop]
    # print(fn)
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

    # Find index closest to interpolated best-track center or 2-km radar center
    distance_arr = distance_calculator(x, y,(desired_lon,desired_lat))

    abs_dist_arr = (abs(distance_arr))

    min_dist_ind = np.where(abs_dist_arr == np.nanmin(abs_dist_arr))

    # Slice arrays to within 30 indices of desired center

    sliced_lon_grid = x[min_dist_ind[0][0]-30:min_dist_ind[0][0]+30,min_dist_ind[1][0]-30:min_dist_ind[1][0]+30]

    sliced_lat_grid = y[min_dist_ind[0][0]-30:min_dist_ind[0][0]+30,min_dist_ind[1][0]-30:min_dist_ind[1][0]+30]

    sliced_precip = precip[min_dist_ind[0][0]-30:min_dist_ind[0][0]+30,min_dist_ind[1][0]-30:min_dist_ind[1][0]+30]

    return np.asarray([sliced_precip,sliced_lon_grid,sliced_lat_grid])

def imerg_parallel_slicing(filepaths_imerg,desired_lon,desired_lat,iteration_length_imerg):
    from multiprocessing import Pool
    pool_imerg = Pool()
    result_slicing = pool_imerg.map(imerg_splat_slicing, ((case_loop,filepaths_imerg,desired_lon,desired_lat) for case_loop in range(iteration_length_imerg)))
    # return np.asarray(result_slicing), result_slicing[0]['latitude'],result_slicing[0]['longitude']
    return result_slicing

def imerg_for_mrms(start_str,end_str,desired_lon,desired_lat,problematic_times_arr):

    # Import libraries
    import pandas as pd
    import datetime as dt
    import os
    import h5py
    import numpy as np
    import sys
    sys.path.insert(0, '/Users/acheung/TC_RI_P1/scripts/Useful Functions/')
    from distance import distance_calculator
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import xarray as xr
    import multiprocessing

    # Enter time and Location

    # start_str = "2020-11-09 00:00:00"
    # end_str = "2020-11-12 00:00:00"

    times_pd = pd.date_range(start=start_str,end=end_str, freq='30min')
    times_pd = times_pd[0:-1]
    # desired_lon = -81.38
    # desired_lat = 28.54

    filepaths_imerg = []
    for i in times_pd: # Download all the desired IMERG Images
        # If IMERG file does not exist, download it
        # The problematic times are offsetted because MRMS timestamp is end of time period, while IMERG is beginning.
        if (len(np.where(i == problematic_times_arr - dt.timedelta(minutes=30))[0]
              ) > 0) | (len(np.where(i + dt.timedelta(minutes=30) == problematic_times_arr - dt.timedelta(minutes=30))[0]) > 0) : # Skip times that MRMS does not have
            print('Skipped ' + str(i)+ ' in IMERG for MRMS')
            continue
        year = i.year
        day_of_year = i.strftime('%j')
        day_stripped = i.strftime('%Y%m%d')
        time_stripped = i.strftime('%H%M%S')
        min_of_day = str(i.hour*60 + i.minute).zfill(4)
        end_time_stripped = (i+dt.timedelta(minutes=29,seconds=59)).strftime('%H%M%S')

        IMERG_file_path = '/Users/acheung/data/IMERG/'+str(year)+'/'+day_of_year+'/'+\
                '3B-HHR.MS.MRG.3IMERG.'+day_stripped+'-S'+time_stripped+'-E'+end_time_stripped+\
                '.'+min_of_day+'.V07B.HDF5'  

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
        filepaths_imerg.append(IMERG_file_path)
        
        
        #---------------------------------------------------------------------------------------------------------
#     def imerg_download(i,filepaths_dict,times_pd): # Download all the desired IMERG Images
#         import os
#         # If IMERG file does not exist, download it
#         time_now = times_pd[i]
#         year = time_now.year
#         day_of_year = time_now.strftime('%j')
#         day_stripped = time_now.strftime('%Y%m%d')
#         time_stripped = time_now.strftime('%H%M%S')
#         min_of_day = str(time_now.hour*60 + time_now.minute).zfill(4)
#         end_time_stripped = (time_now+dt.timedelta(minutes=29,seconds=59)).strftime('%H%M%S')

#         IMERG_file_path = '/Users/acheung/data/IMERG/'+str(year)+'/'+day_of_year+'/'+\
#                 '3B-HHR.MS.MRG.3IMERG.'+day_stripped+'-S'+time_stripped+'-E'+end_time_stripped+\
#                 '.'+min_of_day+'.V07B.HDF5'  

#     # If day directory does not exist, make day directory
#         if os.path.exists('/Users/acheung/data/IMERG/'+str(year)) == False:
#             os.mkdir('/Users/acheung/data/IMERG/'+str(year))

#         if os.path.exists('/Users/acheung/data/IMERG/'+str(year)+'/'+day_of_year) == False:
#             os.mkdir('/Users/acheung/data/IMERG/'+str(year)+'/'+day_of_year)

#         if os.path.exists(IMERG_file_path) == False:
#             url_desired = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/"\
#                 +str(year)+'/'+day_of_year+'/'+'3B-HHR.MS.MRG.3IMERG.'+day_stripped+'-S'+time_stripped+\
#                 '-E'+end_time_stripped+'.'+min_of_day+'.V07B.HDF5'
#             os.chdir('/Users/acheung/data/IMERG/'+str(year)+'/'+day_of_year)

#             os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies '
#                       + url_desired)
#         filepaths_dict[i] = IMERG_file_path

#     # if __name__ == "__main__": # Run multiprocesses
#     manager_3 = multiprocessing.Manager()
#     filepaths_dict = manager_3.dict()
#     jobs = []
#     for i in range(len(times_pd)): # Download all the desired MRMS Images
#         p = multiprocessing.Process(target=imerg_download, args=(i, filepaths_dict,times_pd))
#         jobs.append(p)
#         p.start()

#     for proc in jobs:
#         proc.join()

#     filepaths = filepaths_dict.values()

            
        #---------------------------------------------------------------------------------------------------------


    #-------------------------------------------------------------------------------------------------------

    # this helper function is needed because map() can only be used for functions
    # that take a single argument (see http://stackoverflow.com/q/5442910/1461210)

    iteration_length_imerg = len(filepaths_imerg)
    
    result_slicing= imerg_parallel_slicing(filepaths_imerg,desired_lon,desired_lat,iteration_length_imerg)
    imerg_array = np.asarray(result_slicing)
    imerg_total_precip = imerg_array[:,0]
    sliced_lat_grid = imerg_array[0,2]
    sliced_lon_grid = imerg_array[0,1]

    # Sum up precipitation (IMERG data is half-hour mm/h rain rate)
    total_precip = np.sum(np.asarray(imerg_total_precip),axis=0) * 0.5

    da = xr.DataArray(
        data=total_precip,
        dims=["lon", "lat"],
        coords=dict(
            lon=(["lon", "lat"], sliced_lon_grid),
            lat=(["lon", "lat"], sliced_lat_grid),
            time='start = ' + start_str + ', end = ' + end_str,
        ),
        attrs=dict(
            description="IMERG Precip Total",
            units="mm",
        ),
    )
    
    if os.path.exists('/Users/acheung/data/intermediates/start = ' + start_str + ', end = ' + end_str + ' imerg.nc')==True:
        os.remove('/Users/acheung/data/intermediates/start = ' + start_str + ', end = ' + end_str + ' imerg.nc')
    da.to_netcdf('/Users/acheung/data/intermediates/start = ' + start_str + ', end = ' + end_str + ' imerg.nc')