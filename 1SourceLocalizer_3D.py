# -*- coding: utf-8 -*-
"""
1 Source Localization-3D

Note: This script generates its own count rates to compare. Replace with actual data for use.
"""
import numpy as np
from scipy.optimize import minimize

# Detector locations

ndet = 8
det_pos = np.zeros((ndet,3)) # x,y,z

# half width and length of table
Lx = 80
Ly = 80
# height of lower detectors
H1 =  6  # detector will be above the table where src are placed
H2 =  60 # height of upper detectors
det_pos[0,:] = [0 ,0 ,H1]
det_pos[1,:] = [Lx,0 ,H1]
det_pos[2,:] = [Lx,Ly,H1]
det_pos[3,:] = [0 ,Ly,H1]
det_pos[4,:] = [0 ,0 ,H2]
det_pos[5,:] = [Lx,0 ,H2]
det_pos[6,:] = [Lx,Ly,H2]
det_pos[7,:] = [0 ,Ly,H2]



def cr_fun(src_pt, src_strength, det_pos, verbose=False):
    """
    Generates count rates based on source strength and position, and detector position
    ASSUMPTIONS: count rate is proportional to source strength (i.e. if strength doubles, so does count rate)
    ASSUMPTIONS: count rate us inversely proportional to r^1.5
    
    """
    if verbose:
        print('src pt =',src_pt)
        print('src S  =',src_strength)
    ndet = det_pos.shape[0]
    # count rates
    cr = np.zeros(ndet)
    for idet in range(ndet):
        # compute distance between src and det
        dist = np.linalg.norm(src_pt - det_pos[idet,:])
        cr[idet] = src_strength / dist**1.5
        if verbose:
            print('det#  =',idet)
            print('dist  =',dist)
            print('C.R.  =',cr[idet])
    
    return cr        


# generates predicted count data based on the user defined grid
def count_grid(src_strength,src_pt_arr, xx):
    cr_arr = np.zeros((xx.shape[0],ndet))
    for isrc in range(src_pt_arr.shape[0]):
        src_pt = src_pt_arr[isrc,:3]
        if len(src_pt_arr[1]) == 4:
            src_strength = src_pt_arr[isrc,3]
        cr_arr[isrc,:] = cr_fun(src_pt, src_strength, det_pos)
    
    return cr_arr
        

# find the error between true count rate and count rate at each grid point
def err_grid(cr_arr, cr_true, ns):

   err = np.zeros(nx*ny*nz*ns)
   for idet in range(ndet):
       err += (cr_arr[:,idet] - cr_true[idet])**2
   err = np.sqrt(err)
   err_reshaped = np.reshape(err,(nx,ny,nz,ns))
   
   return err_reshaped   
        
# initial guess for source position based on mesh grid
def init_guess(err_reshaped):
    # find index of smallest error
    idx = np.where(err_reshaped==err_reshaped.min())
    # extract x y z
    i_min = idx[0][0]
    j_min = idx[1][0]
    k_min = idx[2][0]
    x_min = x[i_min]
    y_min = y[j_min]
    z_min = z[k_min]
    if len(idx) == 4:
        src_min = idx[3][0]
        s_min = s[src_min]
        return x_min, y_min, z_min, s_min
    else:
        print(np.shape(idx))
        print(x_min,y_min,err_reshaped[i_min,j_min],err_reshaped.min())
        
        return x_min, y_min, z_min


# optimization for x, y, z, and source
def opt_func3D(x_opt):
    p = x_opt[0:3]
    src_strength = x_opt[3]
    cr_guess = cr_fun(p, src_strength, det_pos)
    err = np.linalg.norm(cr_guess-cr_true)
    return err

# build rough grid for initial position guess
# grid on table (user can change this)
nx, ny, nz = 50, 50, 10

# generate the grid
# NOTE: grid is larger than actual table for plotting purposes
x = np.linspace(-Lx,2*Lx,nx)
y = np.linspace(-Ly,2*Ly,ny)
z = np.linspace(H1+10, H2-10, nz)
X, Y, Z = np.meshgrid(x,y,z)

# make each mesh grid into an nx*ny*nz,1 array shape
xx = np.reshape(X,(-1,1))
yy = np.reshape(Y,(-1,1))
zz = np.reshape(Z,(-1,1))

# combine into an nx*ny*nz,3 shape where a given row shows an x,y,z point on our mesh grid
src_pt_arr = np.zeros((nx*ny*nz,3))
src_pt_arr[:,0] = xx.ravel()
src_pt_arr[:,1] = yy.ravel()
src_pt_arr[:,2] = zz.ravel()


# initialize problem and find true count rate
np.random.seed(12345)
x_true, y_true, z_true = np.random.random(3)
Lz = 50
x_true *= Lx
y_true *= Ly
z_true *= Lz
#print(x_true,y_true, z_true)
src_strength = np.random.random(1)
src_strength = 0.63
#print(src_strength)
cr_true = cr_fun(np.array([x_true,y_true,z_true]), src_strength, det_pos,verbose=False)
#print(cr_true)

# grid on table
nx, ny, nz, ns = 10, 10, 10, 10  #number of grid points x,y,z,source
x = np.linspace(0,Lx,nx)
y = np.linspace(0,Ly,ny)
z = np.linspace(0,Lz,nz)
s = np.linspace(0,1,ns)
X, Y, Z, S = np.meshgrid(x,y,z,s, indexing='ij')
xx = np.reshape(X,(-1,1))
yy = np.reshape(Y,(-1,1))
zz = np.reshape(Z,(-1,1))
ss = np.reshape(S,(-1,1))

# make the source array for points in space and strength
src_pt_arr = np.zeros((nx*ny*nz*ns,4))
src_pt_arr[:,0] = xx.ravel()
src_pt_arr[:,1] = yy.ravel()
src_pt_arr[:,2] = zz.ravel()
src_pt_arr[:,3] = ss.ravel()
# determine counts
cr_arr = count_grid(src_strength,src_pt_arr, xx)
# determine error
err_reshaped = err_grid(cr_arr, cr_true, ns)

x_min, y_min, z_min, s_min = init_guess(err_reshaped)

x_init = np.array([x_min,y_min,z_min,s_min])
p_init = x_init[0:3]
src_strength = x_init[3]
cr_init = cr_fun(p_init, src_strength, det_pos)
err = np.linalg.norm(cr_init-cr_true)
print(f"Best Grid Position at {x_init[0:3]}, Source Strength = {x_init[3]}, loss = {err}")
res = minimize(opt_func3D, x_init, method="BFGS", tol=1e-6)
print("Source Position")
print(res.x[0:3])
print("Source Strength")
print(res.x[3])
#print(res.fun)
