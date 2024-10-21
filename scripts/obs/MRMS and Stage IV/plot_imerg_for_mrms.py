def imerg_for_mrms(start_str,end_str,desired_lon,desired_lat):

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
    from metpy.plots import ctables
    cmap = ctables.registry.get_colortable('precipitation')
    import xarray as xr
    import multiprocessing
    # Enter time and Location

    # start_str = "2020-11-09 00:00:00"
    # end_str = "2020-11-12 00:00:00"

    times_pd = pd.date_range(start=start_str,end=end_str, freq='30min')
    times_pd = times_pd[0:-1]
    # desired_lon = -81.38
    # desired_lat = 28.54

    filepaths = []
    for i in times_pd: # Download all the desired IMERG Images
        # If IMERG file does not exist, download it

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
        filepaths.append(IMERG_file_path)

    def process_imerg(case_loop,imerg_total_precip,filepaths,desired_lon,desired_lat,sliced_lon_arr,sliced_lat_arr):
        import numpy as np
        import xarray as xr
        import sys
        sys.path.insert(0, '/Users/acheung/TC_RI_P1/scripts/Useful Functions/')
        from distance import distance_calculator
        fn = filepaths[case_loop]
        print(fn)
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

        imerg_total_precip[case_loop] = sliced_precip

        if case_loop == 0: # Only grab lat/lon array once for plotting later
            sliced_lat_arr[case_loop] = sliced_lat_grid
            sliced_lon_arr[case_loop] = sliced_lon_grid


    if __name__ == "__main__": # Run multiprocesses
        manager = multiprocessing.Manager()
        imerg_total_precip = manager.dict()
        sliced_lat_arr = manager.dict()
        sliced_lon_arr = manager.dict()
        jobs = []
        for i in range(len(filepaths)):
            p = multiprocessing.Process(target=process_imerg, args=(i, imerg_total_precip,filepaths,desired_lon,desired_lat,
                                                                   sliced_lon_arr,sliced_lat_arr))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

    # Sum up precipitation (MRMS data is hourly QPE)
    total_precip = np.sum(np.asarray(imerg_total_precip.values()),axis=0) * 0.5

    sliced_lat_grid = sliced_lat_arr.values()[0]
    sliced_lon_grid = sliced_lon_arr.values()[0]

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

    da.to_netcdf('/Users/acheung/data/intermediates/start = ' + start_str + ', end = ' + end_str + ' imerg.nc')