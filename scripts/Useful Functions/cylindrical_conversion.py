
# Draw circles from a center point

def circle(center,radius, NOP):
    """
    Parameters
    ----------
    
    x_y_grid_to_cylindrical takes data on an evenly spaced x-y grid and interpolates to 
    cylindrical coordinates.
    :radius_min: Indices of center point as a list [ind1,ind2]
    :radius: Circle of current radius
    :NOP: Number of points in each radius
    
    """
    # Import Librariues
    import numpy as np
    
    # Create an array of azimuths based on number of points (NOP) desired
    THETA=np.linspace(0,2*np.pi,NOP)
    
    # Create an array of radii
    RHO=np.ones((1,NOP))*radius
    
    # Calculate X, Y based on azimuth
    X = RHO * np.cos(THETA)
    Y = RHO * np.sin(THETA)
   
    # Align values to a center point provided above
    X=X+center[0]
    Y=Y+center[1]
    
    # Return X, Y, and azimuths of circle drawn
    return X[0],Y[0],THETA

def x_y_grid_to_cylindrical(center_inds,radius_min,radius_max,radius_interval, NOP,x_grid,y_grid,data):
    
    """
    Parameters
    ----------
    
    x_y_grid_to_cylindrical takes data on an evenly spaced x-y grid and interpolates to 
    cylindrical coordinates. Returns cylindrical data, azimuth array, and radius array.

    :center_inds: Indices of center point as a list [ind1,ind2]
    :radius_min: Minimum radius to be interpolated (based on inputted grid indices)
    :radius_min: Maximum radius to be interpolated (based on inputted grid indices)
    :radius_interval: Interval between radius points
    :NOP: Number of points in each radius
    :x_grid: The x-grid of the data
    :y_grid: The y-grid of the data
    :data: The data on the x-y grid to be interpolated. MUST be an array, NOT DataArray or other formats
    
    """ 
    # Import Libraries
    from cylindrical_conversion import circle
    from scipy.interpolate import RegularGridInterpolator
    import numpy as np
    
    # Create an array of radius we want to draw circles for
    rad_arr = np.arange(radius_min,radius_max + 0.1,radius_interval).astype(int)
    
    # Create an array of zeros with size of (what radii?, number of points in a radius)
    xcirc_all = np.zeros((len(rad_arr),NOP))
    ycirc_all = np.zeros((len(rad_arr),NOP))
    
    # Loop over every radius to draw a circle for each one
    for jc_ind in range(len(rad_arr)): # this is for radial spacing of 1 km from 1-200 km
        jc = rad_arr[jc_ind] # Call the radius for this index
        xcirc,ycirc,THETA = circle(center_inds,jc,NOP) # use the circle function to plot a circle, THETA is in math angles
        
        # Overwrite the zeros array with all the circles drawn
        xcirc_all[jc_ind,:]=xcirc # put x- and y- coordinates of all circles in one variable
        ycirc_all[jc_ind,:]=ycirc
    
    interp = RegularGridInterpolator((y_grid, x_grid), data,bounds_error=False,fill_value=np.nan)
    cylindrical_data = interp((xcirc_all,ycirc_all))
    
    # Return the cylindrical data, array of azimuths (MATH ANGLES, NOT METEO ANGLE), and radius array
    return cylindrical_data, THETA, rad_arr

# Interpolate to equal distance grid

def interp_to_equal_grid(original_lon_arr,original_lat_arr,data,dx,dy): # input as dataframes. dx, dy are ints (units: km)
    
    """
    Parameters
    ----------
    
    interp_to_equal_grid takes data on an unevenly spaced lat/lon grids and interpolates to 
    a evenly spaced lat/lon grid.

    :original_lon_arr: Original longitude grid (unevenly spaced)
    :original_lat_arr: Original latitude grid (unevenly spaced)
    :data: Original data (unevenly spaced)
    :dx: the zonal spacing desired between grid points in km
    :dy: the meridional spacing desired between grid points in km

    Returns: 
    :lon_pd: An array of longitude values that are evenly spaced by desired dx
    :lat_pd: An array of latitude values that are evenly spaced by desired dy
    """ 
    
    # Import Libraries
    import pandas as pd
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    # Create equal distance grid
    Re = 6371 # km

    dtheta = dy/Re
    dtheta_deg = dtheta * 180/np.pi # change in lat degrees per dy desired

    lat_dx_accom = np.arange(pd.DataFrame(original_lat_arr)[0].values[0],pd.DataFrame(original_lat_arr)[0].values[-1],dtheta_deg)
    x_pd = pd.DataFrame(original_lon_arr)
    x_pd.index = pd.DataFrame(original_lat_arr)[0].values

    lon_list = []
    lat_list = []
    for i in range(len(lat_dx_accom)):
        curr_lons = x_pd.iloc[0]
        lat_now = lat_dx_accom[i]
        small_circle = 2 * np.pi * Re * np.cos(lat_now * (np.pi/180))
        dx_at_lat = (small_circle/360) # km per 1 degree
        delta_lambda = dx / dx_at_lat # degree for desired dx value
    #     print(delta_lambda)
        curr_lon_arr = np.arange(x_pd.iloc[0].values[0]+0.01,x_pd.iloc[0].values[-1],delta_lambda)
        lon_list.append(curr_lon_arr)
        lat_list.append(lat_dx_accom)
        

    interp = RegularGridInterpolator((original_lat_arr[:,0],original_lon_arr[0,:]), data)

    lon_pd = pd.DataFrame(lon_list).dropna(axis=1)
    lat_pd = pd.DataFrame(lat_list).transpose()

    if lon_pd.columns.max() > lat_pd.columns.max():
        lon_pd = lon_pd[lat_pd.columns]
    elif lon_pd.columns.max() < lat_pd.columns.max():
        lat_pd = lat_pd[lon_pd.columns]
    eq_dist_data = interp((lat_pd, lon_pd))
    return lon_pd,lat_pd,eq_dist_data