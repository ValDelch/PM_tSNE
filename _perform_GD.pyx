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
from libc.math cimport pow, int
cimport libc.math as math

import scipy.signal
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
color = np.array(['red','blue','green','orange','black','pink','yellow','brown','purple','grey'])


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


cdef double[:] _kernel_2(double[:,:,] z):
    """
    This function correspond to the filter function in the convolution product
    Blabla
    """
    
    cdef:
        int n = z.shape[0]
        double[:] response = np.empty(n)
    
    for i in range(n):
        response[i] = 1.0 / (1.0 + (math.pow(z[i,0], 2.0) + math.pow(z[i,1], 2.0)))
    
    return response


cdef double[:,:] _compute_Repu_NGP(double[:,:] Y, int M, int n):
    """
    Blabla
    """

    cdef:
        int i, j, _i, _j
        double xmax, ymax, xmin, ymin, xy
        double delta_x, delta_y
        double Z = 0.0
        double factor_x, factor_y
        double[:] coord_x  = np.empty((M), dtype=DTYPE)
        double[:] coord_y  = np.empty((M), dtype=DTYPE)
        int[:] idx         = np.empty((n), dtype=DTYPE_int)
        int[:] idy         = np.empty((n), dtype=DTYPE_int)
        int[:,:] hist      = np.zeros((M, M), dtype=DTYPE_int)
        double[:,:] values = np.empty((M,M), dtype=DTYPE)
        double[:,:] pot    = np.empty((M,M), dtype=DTYPE)
        double[:,:] dY     = np.empty((n,2), dtype=DTYPE)

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
    xmax = xmax + delta_x ; xmin = xmin - 1.01*delta_x
    delta_x = (xmax - xmin) / (M - 1)
    
    delta_y = (ymax - ymin) / (M - 1)
    ymax = ymax + delta_y ; ymin = ymin - 1.01*delta_y
    delta_y = (ymax - ymin) / (M - 1)
    
    for i in range(M):
        coord_x[i] = xmin + delta_x * i
        coord_y[i] = ymin + delta_y * i

    # Find the closest point of the grid for each instance
    idx = np.searchsorted(np.asarray(coord_x[:]), np.asarray(Y[:,0]), side='right').astype(DTYPE_int) - 1
    idy = np.searchsorted(np.asarray(coord_y[:]), np.asarray(Y[:,1]), side='right').astype(DTYPE_int) - 1

    # Make the 2D histogram (density map)
    for i in range(n):
        hist[idx[i], idy[i]] += 1

    values = _kernel(coord_x, coord_y)
    pot = scipy.signal.fftconvolve(hist, values, mode='same')
    
    for i in range(n):
        Z = Z + pot[idx[i], idy[i]]
    
    factor_x = pow(Z * delta_x, -1.0)
    factor_y = pow(Z * delta_y, -1.0)
    for i in range(n):
        _i = idx[i]
        _j = idy[i]

        dY[i,0] = (pot[_i+1, _j] - pot[_i-1, _j]) * factor_x
        dY[i,1] = (pot[_i, _j+1] - pot[_i, _j-1]) * factor_y

    return dY


cdef np.ndarray[np.double_t, ndim=2] _compute_Repu_NGP_2(double[:,:] Y, int M, int n):
    """
    This function compute the repulsive term using a convolution product
    """

    cdef:
        int i
        double boundary, dx, Z
        double[:,:] hist, value
        np.ndarray[np.double_t, ndim=1] coord
        np.ndarray[np.int_t, ndim=1] idx, idy
        np.ndarray[np.double_t, ndim=2] pot
        np.ndarray[np.double_t, ndim=2] dY = np.zeros((n, 2))
        np.ndarray[np.double_t, ndim=3] grad
        
    # Get boundaries and grid's resolution
    boundary = np.max(np.abs(Y))

    coord = np.linspace(-1.0 * boundary, 1.0 * boundary, M, endpoint=True)
        
    idx = np.searchsorted(coord[:], Y[:,0], side='left')
    idy = np.searchsorted(coord[:], Y[:,1], side='left')
    
    hist = np.zeros((M,M))
        
    for i in range(n):
        hist[idx[i],idy[i]] += 1
        
    dx = coord[1] - coord[0]
            
    # Compute the grid point's coordinates
    value = np.transpose(np.meshgrid(coord, coord)).reshape(-1, 2)
    
    # convolution product
    value = np.reshape(_kernel_2(value), np.squeeze(hist).shape)
    
    pot = scipy.signal.fftconvolve(hist, value, mode='same')
    
    # Compute the gradient on the grid
    grad = np.gradient(pot) / np.float64(dx)

    dY[:,0] = grad[0,idx[:],idy[:]]
    dY[:,1] = grad[1,idx[:],idy[:]]

    Z = np.sum(pot[idx[:],idy[:]])
    
    return dY / (Z / 2.0)


cdef double[:,:] _compute_Repu_NGP_3(double[:,:] Y, int M, int n):
    """
    This function compute the repulsive term using a convolution product
    """

    cdef:
        int i
        double boundary, dx, Z
        double[:] coord  = np.empty((M), dtype=DTYPE)
        np.ndarray[np.double_t, ndim=3] grad
        int[:] idx         = np.empty((n), dtype=DTYPE_int)
        int[:] idy         = np.empty((n), dtype=DTYPE_int)
        int[:,:] hist     = np.zeros((M, M), dtype=DTYPE_int)
        double[:,:] value = np.empty((M,M), dtype=DTYPE)
        double[:,:] pot    = np.empty((M,M), dtype=DTYPE)
        double[:,:] dY     = np.empty((n,2), dtype=DTYPE)
        
    # Get boundaries and grid's resolution
    boundary = np.max(np.abs(Y))

    coord = np.linspace(-1.0 * boundary, 1.0 * boundary, M, endpoint=True)
        
    idx = np.searchsorted(coord[:], Y[:,0], side='left').astype(DTYPE_int)
    idy = np.searchsorted(coord[:], Y[:,1], side='left').astype(DTYPE_int)
        
    for i in range(n):
        hist[idx[i],idy[i]] += 1
        
    dx = coord[1] - coord[0]
            
    # Compute the grid point's coordinates
    value = np.transpose(np.meshgrid(coord, coord)).reshape(-1, 2)
    
    # convolution product
    value = np.reshape(_kernel_2(value), np.squeeze(hist).shape)
    
    pot = scipy.signal.fftconvolve(hist, value, mode='same')
    
    # Compute the gradient on the grid
    grad = np.gradient(pot) / np.float64(dx)

    for i in range(n):
        dY[i,0] = grad[0,idx[i],idy[i]]
        dY[i,1] = grad[1,idx[i],idy[i]]

        Z = Z + pot[idx[i],idy[i]]
    
    for i in range(n):
        dY[i,0] = dY[i,0] / (Z / 2.0)
        dY[i,1] = dY[i,1] / (Z / 2.0)
    
    return dY


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
        dY_repu = _compute_Repu_NGP_3(Y, M, n)
        print(np.abs(np.asarray(_compute_Repu_NGP_2(Y, M, n)) - np.asarray(dY_repu)) / np.asarray(dY_repu) * 100.0)
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
        
        if l % 5 == 0:
            y = np.load('/home/valentin/ownCloud/Personnel/Memoire/test_PM_2.0/MNIST_labels.npy', allow_pickle=True).astype(np.int)
            plt.scatter(Y[:,0], Y[:,1], c=color[y[:]], s=3)
            plt.savefig('./images/Y_'+str(l)+'.png')
            plt.close('all')

    return Y


cpdef gradientDescent(int n, double[:] data, int[:] indices, int[:] indptr, double coeff, char* grid_meth, 
                      double eta, double early_ex, double initial_mom, double final_mom, double min_gain, 
                      int stop_early, int n_iter):
    """
    Blabla
    """
    
    return _gradientDescent(n, data, indices, indptr, coeff, grid_meth, eta, early_ex, initial_mom, 
                            final_mom, min_gain, stop_early, n_iter)
