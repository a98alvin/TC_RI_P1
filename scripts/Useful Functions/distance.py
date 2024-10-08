import numpy as np


def distance_calculator(lons, lats,destination):
    """ Calculates distance of every point in 2-D lat/lon array to a single lat/lon point
    
    Parameters
    ----------
    lons (array-like): 2-D longitude array
    lats (array-like): 2-D latitude array
    destination (tuple): (lon, lat) of point to calculate distance from
    
    Returns
    -------
    distance (np.array): 2-D array of distances (km) to destination point
    """
    lon1, lat1 = lons, lats
    lon2, lat2 = destination
    # The haversine formula calculation for distance between two lat/lon points
    radius = 6371. # km
    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2.) * np.sin(dlat/2.) + np.cos(np.radians(lat1)) \
        * np.cos(np.radians(lat2)) * np.sin(dlon/2.) * np.sin(dlon/2.)
    c = 2. * np.arctan2(np.sqrt(a), np.sqrt(1.-a))
    distance = radius * c

    return distance

def pointdist_calc(Lat1,Lon1,Lat2,Lon2):
    
    """ Calculates distance between two lat/lon points
    
    Parameters
    ----------
    Lat1: Point 1 Latitude (degrees)
    Lon1: Point 1 Longitude (degrees)
    Lat2: Point 2 Latitude (degrees)
    Lon2: Point 2 Longitde (degrees)
    
    Returns
    -------
    distance (value): Distance between the two points
    """
    from math import sin, cos, sqrt, atan2, radians
    
    # approximate radius of earth in km
    R = 6371
    
    lat1 = radians(Lat1)
    lon1 = radians(Lon1)
    lat2 = radians(Lat2)
    lon2 = radians(Lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    pointdistance = R * c
    
    return pointdistance

def get_angle(lon1,lat1,lon2,lat2):
    import math
    import numpy as np
    lon1 = lon1*(np.pi/180)
    lat1 = lat1*(np.pi/180)
    lon2 = lon2*(np.pi/180)
    lat2 = lat2*(np.pi/180)
    
    delta_lambda = lon2-lon1
    dx = np.sin(delta_lambda)*np.cos(lat2)
    dy = (np.cos(lat1)*np.sin(lat2)) - (np.sin(lat1)* np.cos(lat2) * np.cos(delta_lambda))
    
    return np.arctan2(dy,dx) * 180 / np.pi

def math_to_meteo_angle(math_angle):
    meteo_angle_now_heading = 90 - math_angle
    if math_angle > 90:
        meteo_angle_now_heading = meteo_angle_now_heading + 360
    return meteo_angle_now_heading