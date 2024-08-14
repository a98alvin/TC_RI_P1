% This script reads sample data from a WRF output file and demonstrates
% how to perform an azimuthal average or perform spectral filtering in the
% azumthal direction. The example uses surface pressure perturbation (defined
% as a difference from some arbitrary base solution) as an example.

close all; clear all;

% Read in test data from file into an array called data
ifile = ['/wave/Users/poterjoy/YSTONE_ARCHIVE/Matlab/functions/wrfout'];

function r = cylindrical_fft_test.m(nc_file,center_i_index,center_j_index,radius_of_subdomain,n_slices,radii_interval) % a netcdf file

% data = get_netcdf_var(ifile,'P');
data = nc_file

% Grab surface model level data (IMERG is only 2D)
% data = data(:,:,1);

% Get dimensions of array
[IM,JM,KM] = size(data); 

% Parameters to specify
IO = center_i_index; % i-index of designated center
JO = center_j_index; % j-index of designated center
RM = radius_of_subdomain;  % radius of subdomain for interpolation
NS = n_slices; % number of slices through cylyndrical domain
dr = radii_interval;   % interval of radii

flag = 0; %     ---- OPTIONS ----
          % 0 <--- return original data interpolated back and forth from
          %        cylyndrical coordinates to original cartesian domain
          % 1 <--- return symmetric part; i.e., the azimuthal average
          % 2 <--- return asymetric part; can be modified to get 
          %        wavenumber 1, 2, etc.

% Interpolate data to an NS x RM cylindrical grid
for s = 1:NS
  for l = 1:RM

    rad = (l-1)*dr;
    thet = 2*pi*(s-1)/NS;

    if s == 1, 
      x = rad + IO;
      y = JO;
    else,
      x = rad*cos(thet) + IO;
      y = rad*sin(thet) + JO;
    end

    % Determine distance between radial point and cartesian grid point
    dx = x - floor(x);   % x-distance from left most point
    dy = y - floor(y);   % y-distance from left most point

    % Save x-y coordinates of each point
    xcoord(s,l) = x;
    ycoord(s,l) = y;

    % Interpolate to vector r(:,:,:)
    r(s,l,:) = dy*(dx*data(floor(x)+1,floor(y)+1,:) + ...
             (1-dx)*data(floor(x),floor(y)+1,:)) + ...
             (1-dy)*(dx*data(floor(x)+1,floor(y),:) + ...
	       (1-dx)*data(floor(x),floor(y),:));
  end
end

%{
% Perform scale separation in azimuthal direction
for k = 1:KM
  for l = 1:RM
    % Get power associated with each wavenumber 
    FT = fft(r(:,l,k));
    if flag == 1,
      % Remove everything except wavenumber 0
      FT(2:end) = 0;
    elseif flag == 2,
      % Remove only wavenumber 0 
      % (can easily modify to select wavenumbers to keep)
      FT(1) = 0;
    end
    % Place back in physical space
    r0(:,l,k) = real(ifft(FT));
  end
end

% Interpolate to original grid: Start with original data 
% and replace points within original domain with filtered 
% solution.

odata = data;
beg_I = 1; end_I = IM;
beg_J = 1; end_J = JM;
jcount = 0;   r = r0;

for j = beg_J:end_J

  icount = 0; 
  jcount = jcount + 1;

  for i = beg_I:end_I

    icount = icount + 1;
    x = i - IO; y = j - JO;
    R = sqrt( x*x + y*y );

    if R < RM-1,

      if y >   0, 
        thet = acos(x/R);
      elseif y <  0, 
        thet = 2*pi - acos(x/R);
      else
        if x >= 0, thet = 0; end
        if x <  0, thet = pi; end
      end

      % Get index of slice
      s = ( thet/(2*pi) )*NS + 1;

      % Get differences from indicies
      R = R + 1;
      dt = s - floor(s);
      dr = R - floor(R);

      % Interpolate to Cartesian grid
      if floor(s) == NS,
        odata(icount,jcount,1:KM) = dr*(dt*r(1,floor(R)+1,:) + ...
               (1-dt)*r(NS,floor(R)+1,:)) + ...
               (1-dr)*(dt*r(1,floor(R),:) + ...
  	         (1-dt)*r(NS,floor(R),:));
      else
        odata(icount,jcount,1:KM) = dr*(dt*r(floor(s)+1,floor(R)+1,:) + ...
               (1-dt)*r(floor(s),floor(R)+1,:)) + ...
               (1-dr)*(dt*r(floor(s)+1,floor(R),:) + ...
  	         (1-dt)*r(floor(s),floor(R),:));
      end

    end % if within cylyndrical subdomain domain
  end % j loop
end % i loop

% Generate figures to show that something was done

% Original data
figure(1)
contourf(data(:,:,1)'), colorbar;

% Averaged or filtered data
figure(2)
contourf(odata(:,:,1)'), colorbar;

%}
