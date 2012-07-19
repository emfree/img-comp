import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp
import math



cdef float pi = math.pi


def zip_with_mul(A, B):
    cdef int size = len(A)
    cdef float acc = 0
    cdef int i
    for i in range(size):
        acc += A[i] * B[i]
    return acc


def gaussian(np.ndarray[np.int_t, ndim=1] nbrs, int pixel):
    cdef int size = nbrs.size
    cdef float mean = 0
    cdef float var = 0
    cdef int k
    for k in range(size):
        mean += nbrs[k]
    mean /= size
    for k in range(size):
        var += (nbrs[k] - mean)**2
    var = max(var, 4)
    cdef float alpha = 1 / sqrt(2 * pi * var)
    cdef float beta = - 0.5 / var
    cdef float acc = 0
    for k in range(256):
        acc += alpha * exp(beta * (k - mean)**2)
    return alpha * exp(beta * (pixel - mean)**2) / acc



def combine_weights(np.ndarray[dtype=np.float_t, ndim = 2] weight_subarray):
    cdef int num_weights, num_nbrs
    num_weights = weight_subarray.shape[1]
    num_nbrs = weight_subarray.shape[0]
    cdef int i, j
    combined_weights = np.zeros(num_weights, dtype = float)
    for i in range(num_nbrs):
        for j in range(num_weights):
            combined_weights[j] += weight_subarray[i, j]
    combined_weights /= num_nbrs
    return combined_weights



def multimodal_gaussian(np.ndarray[np.int_t, ndim=1] nbrs, int pixel):
    ## this one is going to be real slow right now
    cdef int size = nbrs.size
    cdef float var = 4
    cdef float alpha = 1 / (2 * pi * var)
    cdef float beta = - 0.5 / var
    cdef float acc 
    cdef float prob
    cdef int j, k
    for j in range(size):
        acc = 0
        for k in range(256):
            acc += alpha * exp(beta * (k - nbrs[j])**2)
        prob += alpha * exp(beta * (pixel - nbrs[j])**2) / acc
    return prob / size
    



