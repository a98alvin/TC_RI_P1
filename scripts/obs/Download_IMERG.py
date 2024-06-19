#!/usr/bin/env python
# coding: utf-8

# # This is script to download IMERG data
# It works because the home directory is signed into the NASA account. Please set up directory before using this script with the directions in this link: https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Access%20GES%20DISC%20Data%20Using%20wget%20and%20curl

def IMERG_download(year,month,day,hour,minute):
    import os
    import datetime as dt
    time_desired = dt.datetime(year,month,day,hour,minute)
    year = time_desired.year
    day_of_year = time_desired.strftime('%j')
    day_stripped = time_desired.strftime('%Y%m%d')
    time_stripped = time_desired.strftime('%H%M%S')
    min_of_day = str(time_desired.hour*60 + time_desired.minute).zfill(4)
    end_time_stripped = (time_desired+dt.timedelta(minutes=29,seconds=59)).strftime('%H%M%S')
    # If year directory does not exist, make year directory
    if os.path.exists('/Users/acheung/data/IMERG/'+str(year)) == False:
        os.mkdir('/Users/acheung/data/IMERG/'+str(year))

    # If day directory does not exist, make day directory
    if os.path.exists('/Users/acheung/data/IMERG/'+str(year)+'/'+day_of_year) == False:
        os.mkdir('/Users/acheung/data/IMERG/'+str(year)+'/'+day_of_year)

    os.chdir('/Users/acheung/data/IMERG/'+str(year)+'/'+day_of_year)
    url_desired = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/"+str(year)+'/'+day_of_year+'/'+'3B-HHR.MS.MRG.3IMERG.'+day_stripped+'-S'+time_stripped+'-E'+end_time_stripped+'.'+min_of_day+'.V07B.HDF5'
    
    IMERG_file_path = '/Users/acheung/data/IMERG/'+str(year)+'/'+day_of_year+'/'+'3B-HHR.MS.MRG.3IMERG.'+day_stripped+'-S'+time_stripped+'-E'+end_time_stripped+'.'+min_of_day+'.V07B.HDF5'
    
    if os.path.exists(IMERG_file_path) == False: # Won't download files that already exist!
        os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies '+ url_desired)
    
    return IMERG_file_path