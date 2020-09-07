# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=True

"""
This part of the code impement the gradient descent used for PM_tSNE.

More informations in PM_tSNE.py
"""


import numpy as np
cimport numpy as np
from libc.math cimport pow, ceil

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPE_int = np.int32
ctypedef np.int32_t DTYPE_int_t


cdef double[:,:] _compute_Attr(double[:,:] Y, double[:] data, int[:] indices, int[:] indptr, int n, 
                               int no_dims):
    """
    This function compute the attractive term with respect to the k nearest neighbors
    
    ----------
    Parameters
    ----------
    * Y : 
        Blabla.
        
    * data :
        Balbla.
        
    * indices :
        Blabla.
        
    * indptr :
        Blabla.
        
    * n :
        Blabla.
        
    * no_dims :
        Blabla.
        
    -------
    Returns
    -------
    * dY : 
        Blabla.
    """
    
    cdef:
        int i, j, _i, _j
        double qz, diff1, diff2
        double[:,:] dY = np.zeros((n, no_dims), dtype=DTYPE)
        
    for i in range(n):
        _i = indptr[i]
        _j = indptr[i+1]
        
        for j in range(_i, _j):
            diff1 = Y[i,0] - Y[indices[j],0]
            diff2 = Y[i,1] - Y[indices[j],1]
            qz = 4.0 / (1.0 + pow(diff1, 2.0)
                            + pow(diff2, 2.0))
            
            dY[i,0] += data[j] * qz * diff1
            dY[i,1] += data[j] * qz * diff2
            
    return dY


cdef double _kernel(double z_x, double z_y):
    """
    Blabla
    """
    
    cdef:
        double response

    response = 1.0 / (1.0 + pow(z_x, 2.0) + pow(z_y, 2.0))
        
    return response


cdef int[:] _searchsorted(double[:] values, double[:] items):
    
    cdef:
        int i, j, last_idx
        int n_items = items.shape[0]
        int n_values = values.shape[0]
        int[:] out = np.empty((n_items), dtype=DTYPE_int)
        
    for i in range(n_items):
        last_idx = 0
        for j in range(last_idx, n_values):
            if items[i] <= values[j]:
                break
            last_idx = j
        out[i] = last_idx
        
    return out


cdef double[:,:] _compute_Repu_NGP(double[:,:] Y, double coeff, int n, int no_dims):
    """
    Blabla
    """
    
    cdef:
        int i
        int M = ceil(pow(2.0, coeff)) // 2 * int(2) + 1
        double xmax, ymax, xmin, ymin, xy
        double delta_x, delta_y
        double[:] coord_x  = np.empty((M), dtype=DTYPE)
        double[:] coord_y  = np.empty((M), dtype=DTYPE)
        int[:] idx, idy
        int[:,:] hist      = np.zeros((M, M), dtype=DTYPE_int)
        double[:,:] values = np.empty((M, M), dtype=DTYPE) 
        double[:,:] pot
        double[:,:] dY     = np.zeros((n, no_dims), dtype=DTYPE)

    # Edge of the grid
    xmax = Y[0,0] ; xmin = Y[0,0]
    ymax = Y[0,1] ; ymin = Y[0,1]
    for i in range(1,n):
        xy = Y[i,0] 
        if xy < xmin:
            xmin = xy
        elif xy > xmax:
            xmax = xy
            
        xy = Y[i,1]
        if xy < ymin:
            ymin = xy
        elif xy > ymax:
            ymax = xy
            
    delta_x = (xmax - xmin) / (M - 1)
    delta_y = (ymax - ymin) / (M - 1)
    for i in range(M):
        coord_x[i] = xmin + delta_x * i
        coord_y[i] = ymin + delta_y * i
        
    # Find the closest point of the grid for each instance
    idx = _searchsorted(coord_x, Y[:,0])
    idy = _searchsorted(coord_y, Y[:,1])
    
    # Make the 2D histogram (density map)
    for i in range(n):
        hist[idx[i], idy[i]] += 1
    
    for j in range(M):
        for i in range(M):
            values[i,j] = _kernel(coord_x[i], coord_y[j])
    
    return dY
            

cdef double[:,:] _gradientDescent(int n, int no_dims, double[:] data, int[:] indices, int[:] indptr, 
                                  double coeff, char* grid_meth, double eta, double early_ex, 
                                  double initial_mom, double final_mom, double min_gain, int stop_early, 
                                  int n_iter):
    """
    Blabla
    """

    cdef:
        int i, j, l
        double momentum
        double[:] sum_res   = np.zeros((no_dims), dtype=DTYPE)
        double[:,:] dY_tot  = np.empty((n, no_dims), dtype=DTYPE)
        double[:,:] dY_attr = np.empty((n, no_dims), dtype=DTYPE)
        double[:,:] dY_repu = np.empty((n, no_dims), dtype=DTYPE)
        double[:,:] Y       = np.random.randn(n, no_dims).astype(dtype=DTYPE) * 1e-4
        double[:,:] gains   = np.zeros((n, no_dims), dtype=DTYPE)
        double[:,:] iY      = np.zeros((n, no_dims), dtype=DTYPE)
        
    # Early exaggeration
    for i in range(data.shape[0]):
        data[i] *= early_ex
        
    # Main loop
    for l in range(n_iter):
        if l < 20:
            momentum = initial_mom
        else:
            momentum = final_mom

        dY_attr = _compute_Attr(Y, data, indices, indptr, n, no_dims)
        dY_repu = _compute_Repu_NGP(Y, coeff, n, no_dims)
        for i in range(n):
            dY_tot[i,0] = dY_attr[i,0] + dY_repu[i,0]
            dY_tot[i,1] = dY_attr[i,1] + dY_repu[i,1]
        
        sum_res[:] = 0.0
        for i in range(n):
            for j in range(no_dims):
                if dY_tot[i,j] > 0.0 != iY[i,j] > 0.0:
                    gains[i,j] += 0.2
                else:
                    gains[i,j] *= 0.8
                    
                if gains[i,j] < min_gain:
                    gains[i,j] = min_gain
                    
                iY[i,j] = momentum * iY[i,j] - eta * gains[i,j] * dY_tot[i,j]
                Y[i,j] += iY[i,j]
                sum_res[j] += Y[i,j]
                
        for i in range(n):
            for j in range(no_dims):
                Y[i,j] -= sum_res[j]/n
                
        if l == stop_early:
            for i in range(data.shape[0]):
                data[i] /= early_ex
        break
    return Y


cpdef gradientDescent(int n, int no_dims, double[:] data, int[:] indices, int[:] indptr, double coeff, 
                      char* grid_meth, double eta, double early_ex, double initial_mom, double final_mom, 
                      double min_gain, int stop_early, int n_iter):
    """
    Blabla
    """
    
    return _gradientDescent(n, no_dims, data, indices, indptr, coeff, grid_meth, eta, early_ex, initial_mom, 
                            final_mom, min_gain, stop_early, n_iter)
