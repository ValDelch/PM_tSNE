# cython: language_level=3
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
from libc.math cimport pow, int, abs
import scipy.signal


DTYPE = np.double
ctypedef np.double_t DTYPE_t

DTYPE_int = np.int32
ctypedef np.int32_t DTYPE_int_t


cdef double[:,:] _compute_Attr(double[:,:] Y, double[:] data, int[:] indices,
                               int[:] indptr, int n):
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
        
    -------
    Returns
    -------
    * dY : 
        Blabla.
    """
    
    cdef:
        int i, j, _i, _j
        double qz, diff1, diff2
        double[:,:] dY = np.empty((n,2), dtype=DTYPE)
        
    for i in range(n):
        _i = indptr[i]
        _j = indptr[i+1]
        
        diff1 = Y[i,0] - Y[indices[_i],0]
        diff2 = Y[i,1] - Y[indices[_i],1]
        qz = pow(1.0 + pow(diff1, 2.0)
                     + pow(diff2, 2.0), -1.0)
            
        dY[i,0] = data[_i] * qz * diff1
        dY[i,1] = data[_i] * qz * diff2
        
        for j in range(_i+1, _j):
            diff1 = Y[i,0] - Y[indices[j],0]
            diff2 = Y[i,1] - Y[indices[j],1]
            qz = pow(1.0 + pow(diff1, 2.0)
                         + pow(diff2, 2.0), -1.0)
            
            dY[i,0] = dY[i,0] + data[j] * qz * diff1
            dY[i,1] = dY[i,1] + data[j] * qz * diff2
            
        dY[i,0] = 4.0 * dY[i,0]
        dY[i,1] = 4.0 * dY[i,1]
            
    return dY


cdef double[:,:] _kernel(double[:] x, double[:] y):
    """
    Blabla
    """
    
    cdef:
        int i, j
        int M = x.shape[0]
        double[:,:] response = np.empty((M,M), dtype=DTYPE)

    for i in range(M):
        for j in range(M):
            response[j,i] = pow(1.0 + pow(x[j], 2.0) + pow(y[i], 2.0), -1.0)
        
    return response


cdef double[:,:] _compute_Repu_NGP(double[:,:] Y, int M, int n):
    """
    This function compute the repulsive term using a convolution product
    """

    cdef:
        int i, j
        double boundary = 0.0
        double dx
        double Z
        int[:,:] hist = np.zeros((M, M), dtype=DTYPE_int)
        double[:,:] values
        np.ndarray[np.int_t, ndim=1] idx, idy
        np.ndarray[np.double_t, ndim=1] coord
        np.ndarray[np.double_t, ndim=2] pot
        np.ndarray[np.double_t, ndim=2] dY = np.empty((n, 2), dtype=DTYPE)
        np.ndarray[np.double_t, ndim=3] grad
        
    # Get boundaries and grid's resolution
    for i in range(n):
        for j in range(2):
            if abs(Y[i,j]) > boundary:
                boundary = abs(Y[i,j])
                
    dx = (2.0 * boundary) / (M - 1)
    boundary = boundary + dx
    
    coord = np.linspace(-1.0 * boundary, 1.0 * boundary, M, endpoint=True)

    idx = np.searchsorted(coord[:], Y[:,0], side='left') - 1
    idy = np.searchsorted(coord[:], Y[:,1], side='left') - 1
    dx = coord[1] - coord[0]
    
    for i in range(n):
        if abs(coord[idx[i]] - Y[i,0]) > (dx / 2.0):
                idx[i] += 1
        if abs(coord[idy[i]] - Y[i,1]) > (dx / 2.0):
                idy[i] += 1
    
    for i in range(n):
        hist[idx[i],idy[i]] += 1
            
    values = _kernel(coord, coord)
    pot = scipy.signal.fftconvolve(hist, values, mode='same')
    grad = np.gradient(pot) / np.float64(dx)
    
    Z = np.sum(pot[idx[:], idy[:]]) / 2.0
    
    dY[:,0] = grad[0,idx[:],idy[:]]
    dY[:,1] = grad[1,idx[:],idy[:]]
    
    return dY / Z


cdef double[:,:] _compute_Repu_CIC(double[:,:] Y, int M, int n):
    """
    This function compute the repulsive term using a convolution product
    """

    cdef:
        int i
        double boundary = 0.0
        double dx
        double Z = 0
        double[:,:] hist = np.zeros((M, M), dtype=DTYPE)
        double[:,:] values
        np.ndarray[np.int_t, ndim=1] idx, idy
        np.ndarray[np.double_t, ndim=1] coord
        np.ndarray[np.double_t, ndim=2] pot
        np.ndarray[np.double_t, ndim=2] dY = np.zeros((n, 2))
        np.ndarray[np.double_t, ndim=3] grad
        
    # Get boundaries and grid's resolution
    for i in range(n):
        for j in range(2):
            if abs(Y[i,j]) > boundary:
                boundary = abs(Y[i,j])
                
    dx = (2.0 * boundary) / (M - 1)
    boundary = boundary + dx
    
    coord = np.linspace(-1.0 * boundary, 1.0 * boundary, M, endpoint=True)

    idx = np.searchsorted(coord[:], Y[:,0], side='left') - 1
    idy = np.searchsorted(coord[:], Y[:,1], side='left') - 1
    dx = coord[1] - coord[0]
    
    for i in range(n):
        if abs(coord[idx[i]] - Y[i,0]) > (dx / 2.0):
                idx[i] += 1
        if abs(coord[idy[i]] - Y[i,1]) > (dx / 2.0):
                idy[i] += 1
    
    for i in range(n):
        hist[idx[i],idy[i]] += pow(dx, -2.0) * abs((coord[idx[i]+1] - Y[i,0]) * (coord[idy[i]+1] - Y[i,1]))
        hist[idx[i]+1,idy[i]] += pow(dx, -2.0) * abs((coord[idx[i]] - Y[i,0]) * (coord[idy[i]+1] - Y[i,1]))
        hist[idx[i],idy[i]+1] += pow(dx, -2.0) * abs((coord[idx[i]+1] - Y[i,0]) * (coord[idy[i]] - Y[i,1]))
        hist[idx[i]+1,idy[i]+1] += pow(dx, -2.0) * abs((coord[idx[i]] - Y[i,0]) * (coord[idy[i]] - Y[i,1]))
            
    values = _kernel(coord, coord)
    pot = scipy.signal.fftconvolve(hist, values, mode='same')
    grad = np.gradient(pot) / np.float64(dx)

    for i in range(n):
        dY[i,0] = grad[0,idx[i],idy[i]] + ((grad[0,idx[i]+1,idy[i]] - grad[0,idx[i],idy[i]]) / dx) * (Y[i,0] - coord[idx[i]]) + \
                    ((grad[0,idx[i],idy[i]+1] - grad[0,idx[i],idy[i]]) / dx) * (Y[i,1] - coord[idy[i]]) + \
                    ((grad[0,idx[i]+1,idy[i]+1] - grad[0,idx[i],idy[i]]) / (dx**2)) * (Y[i,0] - coord[idx[i]]) * (Y[i,1] - coord[idy[i]])
        dY[i,1] = grad[1,idx[i],idy[i]] + ((grad[1,idx[i]+1,idy[i]] - grad[1,idx[i],idy[i]]) / dx) * (Y[i,0] - coord[idx[i]]) + \
                    ((grad[1,idx[i],idy[i]+1] - grad[1,idx[i],idy[i]]) / dx) * (Y[i,1] - coord[idy[i]]) + \
                    ((grad[1,idx[i]+1,idy[i]+1] - grad[1,idx[i],idy[i]]) / (dx**2)) * (Y[i,0] - coord[idx[i]]) * (Y[i,1] - coord[idy[i]])
        Z = Z + (pot[idx[i],idy[i]] + ((pot[idx[i]+1,idy[i]] - pot[idx[i],idy[i]]) / dx) * (Y[i,0] - coord[idx[i]]) + \
                    ((pot[idx[i],idy[i]+1] - pot[idx[i],idy[i]]) / dx) * (Y[i,1] - coord[idy[i]]) + \
                    ((pot[idx[i]+1,idy[i]+1] - pot[idx[i],idy[i]]) / (dx**2)) * (Y[i,0] - coord[idx[i]]) * (Y[i,1] - coord[idy[i]]))
    
    return dY / (Z / 2.0)


cdef double[:,:] _gradientDescent(int n, double[:] data, int[:] indices, int[:] indptr, double coeff, 
                                  char* grid_meth, double eta, double early_ex, double initial_mom, 
                                  double final_mom, double min_gain, int stop_early, int n_iter):
    """
    Blabla
    """

    cdef:
        int i, j, l
        int M = int(pow(2.0, coeff))
        double momentum
        double[:] sum_res   = np.empty((2), dtype=DTYPE)
        double[:,:] dY_tot  = np.empty((n,2), dtype=DTYPE)
        double[:,:] dY_attr = np.empty((n,2), dtype=DTYPE)
        double[:,:] dY_repu = np.empty((n,2), dtype=DTYPE)
        double[:,:] Y       = np.random.randn(n,2).astype(dtype=DTYPE)
        double[:,:] gains   = np.ones((n,2), dtype=DTYPE)
        double[:,:] iY      = np.zeros((n,2), dtype=DTYPE)

    # Early exaggeration
    for i in range(data.shape[0]):
        data[i] = data[i] * early_ex

    # Main loop
    for l in range(n_iter):
        if l < 20:
            momentum = initial_mom
        else:
            momentum = final_mom

        dY_attr = _compute_Attr(Y, data, indices, indptr, n)
        if grid_meth == 'NGP'.encode('utf-8'):
            dY_repu = _compute_Repu_NGP(Y, M, n)
        else:
            dY_repu = _compute_Repu_CIC(Y, M, n)
        for i in range(n):
            dY_tot[i,0] = dY_attr[i,0] + dY_repu[i,0]
            dY_tot[i,1] = dY_attr[i,1] + dY_repu[i,1]

        sum_res[0] = 0.0 ; sum_res[1] = 0.0
        for i in range(n):
            for j in range(2):
                if dY_tot[i,j] * iY[i,j] < 0.0:
                    gains[i,j] = gains[i,j] + 0.2
                else:
                    gains[i,j] = gains[i,j] * 0.8
                    
                if gains[i,j] < min_gain:
                    gains[i,j] = min_gain
                    
                iY[i,j] = momentum * iY[i,j] - eta * gains[i,j] * dY_tot[i,j]
                Y[i,j] = Y[i,j] + iY[i,j]
                sum_res[j] = sum_res[j] + Y[i,j]

        sum_res[0] = sum_res[0] / n
        sum_res[1] = sum_res[1] / n
        for i in range(n):
            Y[i,0] = Y[i,0] - sum_res[0]
            Y[i,1] = Y[i,1] - sum_res[1]

        if l == stop_early:
            for i in range(data.shape[0]):
                data[i] = data[i] / early_ex

    return Y


cpdef gradientDescent(int n, double[:] data, int[:] indices, int[:] indptr, double coeff, char* grid_meth, 
                      double eta, double early_ex, double initial_mom, double final_mom, double min_gain, 
                      int stop_early, int n_iter):
    """
    Blabla
    """
    
    return _gradientDescent(n, data, indices, indptr, coeff, grid_meth, eta, early_ex, initial_mom, 
                            final_mom, min_gain, stop_early, n_iter)
