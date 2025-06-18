def download_era5_single_level(variable,year_str,month_str,day_str,hour_str,era5_file_path):
    import cdsapi
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [variable],
        "year": [year_str],
        "month": [month_str],
        "day": [day_str],
        "time": [hour_str+":00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    client = cdsapi.Client()
    client.retrieve(dataset, request,era5_file_path)