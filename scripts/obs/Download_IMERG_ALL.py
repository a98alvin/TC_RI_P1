#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import datetime as datetime
from Download_IMERG import IMERG_download

# Identify times/storms needed

desired_basin = 'east_pacific'
# Temporal smooothing?
# Two/four (even number of) IMERG files would have a center at time.
# If no smoothing desired, use a single forward time file, set to zero.
smoothing_times = 2 # delta number of times before or after to evaluate for smoothing. 2 would mean 4 total files

dataset = pd.read_csv('/Users/acheung/data/SHIPS/all_SHIPs_data_combined_'+desired_basin+'.csv')

# Download times interested
times_to_test = pd.to_datetime(dataset.Time)

IMERG_file_path_list = []
IMERG_file_path_forward_list = []
IMERG_file_path_backward_list = []

for t_i in range(len(times_to_test)):
    current_dt = times_to_test.iloc[t_i]
    if smoothing_times > 0:
        forward_now_list = []
        backward_now_list = []
        for t_wind in range(smoothing_times):
            forward_time = current_dt + (datetime.timedelta(minutes=30)*t_wind)
            backward_time = current_dt - (datetime.timedelta(minutes=30)*(t_wind + 1))
#             print(forward_time,backward_time)
            IMERG_file_path_forward = IMERG_download(forward_time.year,forward_time.month,
                                                     forward_time.day,forward_time.hour,
                                                     forward_time.minute)
            forward_now_list.append(IMERG_file_path_forward)

            IMERG_file_path_backward = IMERG_download(backward_time.year,backward_time.month,
                                                     backward_time.day,backward_time.hour,
                                                     backward_time.minute)

            backward_now_list.append(IMERG_file_path_backward)
        IMERG_file_path_backward_list.append(backward_now_list)
        IMERG_file_path_forward_list.append(forward_now_list)
            
    elif smoothing_times == 0:
        IMERG_file_path = IMERG_download(current_dt.year,current_dt.month,current_dt.day,current_dt.hour,current_dt.minute)
        IMERG_file_path_list.append(IMERG_file_path)
        
    print(str(t_i) + ' of ' + str(len(times_to_test))) 
    

