#cython: profile=True
#cython: boundscheck=False

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt
from math import log
from PIL import Image
from heapq import heapify, heappop, heappush
from numpy.lib.stride_tricks import as_strided
import os
from aux import order2



ctypedef float (*nbrfunc)(list, int)



cdef mean(list vals):
    cdef float mean = 0
    cdef int i
    cdef int size = len(vals)
    for i in range(size):
        mean += vals[i]
    return mean / size


cdef float fixed_laplacian(list nbrs, int val):
    beta = - 0.2
    cdef float avg = mean(nbrs)
    cdef float total = 0
    cdef int i
    for i in range(256):
        total += exp(beta * abs(i - avg)) 
    return exp(beta * abs(val - avg)) / total



cdef float variable_laplacian(list nbrs, int val):
    cdef float avg = mean(nbrs)
    cdef float var = sum([(nbr - avg)**2 for nbr in nbrs]) * 1. / len(nbrs)
    cdef float beta
    if var > 0:
        beta = - 1. / sqrt(var / 2)
    else:
        return 0.5 if val == avg else 0.5 / 255
    cdef float total = 0
    cdef int i
    for i in range(256):
        total += exp(beta * abs(i - avg))
    return exp(beta * abs(val - avg)) / total


cdef float mixed_fixed_laplacians(list nbrs, int val):
    cdef float avg = mean(nbrs)
    cdef int nbr
    cdef float beta = -1. / 2.25
    cdef float total = 0
    cdef int i
    for i in range(256):
        for nbr in nbrs:
            total += exp(beta * abs(i - nbr))
    cdef float pred = 0
    for nbr in nbrs:
        pred += exp(beta * abs(val - nbr))
    return  pred / total


cdef float mixed_variable_laplacians(list nbrs, int val):
    cdef float avg = mean(nbrs)
    cdef int nbr
    cdef float var = sum([(nbr - avg)**2 for nbr in nbrs]) * 1. / len(nbrs)
    cdef float beta
    if var > 0:
        beta = - 1. / sqrt(var / 2)
    else:
        return 0.5 if val == avg else 0.5 / 255
    cdef float total
    cdef float pred = 0
    cdef int i
    for nbr in nbrs:
        total = 0
        for i in range(256):
            total += exp(beta * abs(i - nbr))
        pred += exp(beta * abs(val - nbr)) / total
    return  pred / len(nbrs)




cdef float mixed_variable_laplacians2(list nbrs, int val):
    cdef float avg = mean(nbrs)
    cdef int nbr
    cdef float diversity = sum([abs(nbr - avg) for nbr in nbrs]) * 1. / len(nbrs)
    cdef float beta = 1
    if diversity == 0:
        return 0.9 if val == avg else 0.1 / 255
    beta = -1. / diversity
    cdef float total = 0
    cdef int i
    for i in range(256):
        for nbr in nbrs:
            total += exp(beta * abs(i - nbr))
    cdef float pred = 0
    for nbr in nbrs:
        pred += exp(beta * abs(val - nbr))
    return  pred / total



cdef float variable_gaussian(list nbrs, int val):
    cdef float avg = mean(nbrs)
    cdef float var = sum([(nbr - avg)**2 for nbr in nbrs]) * 1. / len(nbrs)
    cdef float beta
    if var > 0:
        beta = - 1. / var
    else:
        return 0.5 if val == avg else 0.5 / 255
    cdef float total = 0
    cdef int i
    for i in range(256):
        total += exp(beta * (i - avg)**2)
    return exp(beta * abs(val - avg)**2) / total



cdef float uniform(list nbrs, int val):
    return 1. / 256



cdef list nbrs(np.ndarray[np.int_t, ndim=2] data, int i, int j, int level, int length, bint skew = False):
    if skew:
        return [data[i - level, j + level], data[i + level, j - level], data[i + level, j + level], data[i - level, j - level]]
    else:
        if i % (length - 1) == 0:
            return [data[abs(i - level), j], data[i, j + level], data[i, j - level]]
        elif j % (length - 1) == 0:
            return [data[i, abs(j - level)], data[i - level, j], data[i + level, j]]
        else:
            return [data[i - level, j], data[i + level, j], data[i, j - level], data[i, j + level]]


cdef apply_func(np.ndarray[np.int_t, ndim=2] data, nbrfunc func, np.ndarray[np.float_t, ndim=2] dest, list metadata):
    cdef int length = data.shape[0]
    cdef int level = length // 2
    cdef int i, j, start
    cdef float val
    while level > 0:
        for i from level <= i < length by 2 * level:
            for j from level <= j < length by 2 * level:
                nbrvals = nbrs(data, i, j, level, length, True)
                val = func(nbrvals, data[i, j])
                dest[i, j] = val
                metadata.append((val, i, j, data[i, j], nbrvals))
        for i from 0 <= i < length by level:
            if i % (2 * level) == 0:
                start = level
            else:
                start = 0
            for j from start <= j < length by 2 * level:
                nbrvals = nbrs(data, i, j, level, length, False)
                val = func(nbrvals, data[i, j])
                dest[i, j] = val
                metadata.append((val, i, j, data[i, j], nbrvals))
        level /= 2



## get huffman code length from distribution
cdef float codelength(np.ndarray[np.float_t] dist, int index):
    cdef int i
    trees = [(dist[i], i == index) for i in range(256)]
    heapify(trees)
    cdef float count = 0
    while len(trees) > 1:
        right, left = heappop(trees), heappop(trees)
        parent = (right[0] + left[0], right[1] or left[1])
        if parent[1]:
            count += 1
        heappush(trees, parent)
    return count

cdef float huffman_variable_laplacian(list nbrs, int val):
    cdef float avg = mean(nbrs)
    cdef float var = sum([(nbr - avg)**2 for nbr in nbrs])
    cdef float beta
    cdef int i
    cdef np.ndarray[np.float_t] dist = np.zeros(256, dtype = float)
    if var == 0:
        return 1. if val == avg else 9.
    else:
        beta = - 1. / sqrt(var / 2)
        for i in range(256):
            dist[i] = exp(beta * abs(val - avg))
        return codelength(dist, val)



cdef rasterscan(np.ndarray[np.int_t, ndim=2] data, nbrfunc func):
    cdef int length = data.shape[0]
    cdef int ls = data.strides[0]
    cdef int ss = data.strides[1]
    cdef int i
    nbr_array = as_strided(data, (length - 1, length - 1, 2, 2), (ls, ss, ls, ss)).reshape(((length - 1)**2, 4))
    output = np.zeros((length-1)**2, dtype = float)
    for i in range(nbr_array.shape[0]):
        output[i] = func(list(nbr_array[i, :-1]), nbr_array[i, -1])
    return output


def approx_codelength(array):
    return -np.sum(np.log2(array[array > 0]))



def predict(img, predictor = 'mvl'):
    data = np.asarray(img, dtype = int)
    dest = np.zeros((513, 513), dtype = float)
    metadata = []
    if predictor == 'vl':
        apply_func(data, variable_laplacian, dest, metadata)
    elif predictor == 'mvl':
        apply_func(data, mixed_variable_laplacians, dest, metadata)
    elif predictor == 'mvl2':
        apply_func(data, mixed_variable_laplacians2, dest, metadata)
    elif predictor == 'vg':
        apply_func(data, variable_gaussian, dest, metadata)
    elif predictor == 'mfl':
        apply_func(data, mixed_fixed_laplacians, dest, metadata)
    return dest, metadata



def show_prediction(pred):
    img = Image.fromarray((255. / pred.max() * pred).astype(np.uint8))
    img.show()

def level_mask(length, level):
    mask = np.zeros((length, length), dtype = bool)
    for i in range(length):
        for j in range(length):
            if (order2(i) == level and order2(j) == level) or (order2(i - j) == level and order2(i + j) == level):
                mask[i, j] = True
    return mask


def show_mask(length, level):
    mask = level_mask(length, level)
    img = Image.fromarray(255 * mask.astype(np.uint8))
    img.show()


def pyfunc(nbrs, val, predictor = 'mvl'):
    if predictor == 'vl':
        return variable_laplacian(nbrs, val)
    elif predictor == 'mvl':
        return mixed_variable_laplacians(nbrs, val)
    elif predictor == 'mvl2':
        return mixed_variable_laplacians2(nbrs, val)
    elif predictor == 'vg':
        return variable_gaussian(nbrs, val)
    elif predictor == 'mfl':
        return mixed_fixed_laplacians(nbrs, val)


