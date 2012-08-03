import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp
import math
from heapq import heapify, heappop, heappush
cimport cpython.array as array


ctypedef float (*hierfunc)(np.ndarray[np.float_t], int)

cdef float pi = math.pi

## some auxiliary functions

cdef float mean(array.array[int] arr):
    cdef float m = 0
    cdef int size = len(arr)
    cdef int i
    for i in range(size):
        m += arr[i]
    return m / size


cdef float median(list array):
    cdef float m
    cdef int size = array.shape[0]
    if size % 2 == 0:
        m = array[size // 2] + array[size // 2 + 1]
        return m / 2
    else:
        return array[size // 2]


cdef float asum(array.array[float] arr):
    cdef float s = 0
    cdef int i
    cdef int size = len(arr)
    for i in range(size):
        s += arr[i]
    return s




## the experts

cpdef float mean_gaussian(int var, array.array[int] nbrs, int pixel):
    cdef array.array[float] dist = array.array('f')
    cdef float m = mean(nbrs)
    cdef float beta = - 0.5 / var
    cdef int i
    for i in range(256):
        dist.append(exp(beta * (i - m)**2))
    return dist[pixel] / asum(dist)

cpdef float median_gaussian(int var, np.ndarray[np.int_t, ndim=1] nbrs, int pixel):
    cdef np.ndarray[np.float_t, ndim=1] dist = np.empty(256, np.float)
    cdef float m = median(nbrs)
    cdef float beta = - 0.5 / var
    cdef int i
    for i in range(256):
        dist[i] = exp(beta * (i - m)**2)
    dist /= asum(dist)
    return dist[pixel]


cpdef float variable_gaussian(np.ndarray[np.int_t, ndim=1] nbrs, int pixel):
    cdef np.ndarray[np.float_t, ndim=1] dist = np.empty(256, np.float)
    cdef float var = 0
    cdef float m = mean(nbrs)
    cdef int i
    cdef int size = nbrs.shape[0]
    for i in range(size):
        var += (nbrs[i] - m)**2
    var /= (size - 1)
    var = max(var, 20)
    cdef float beta = - 0.5 / var
    for i in range(256):
        dist[i] = exp(beta * (i - m)**2)
    dist /= asum(dist)
    return dist[pixel]


cpdef float mean_laplacian(float b, np.ndarray[np.int_t, ndim=1] nbrs, int pixel, bint cl = False):
    cdef np.ndarray[np.float_t, ndim=1] dist = np.empty(256, np.float)
    cdef float m = mean(nbrs)
    cdef int i
    cdef float beta = - 1. / b
    for i in range(256):
        dist[i] = exp(beta * abs(i - m))
    dist /= asum(dist)
    if not cl:
        return dist[pixel]
    else:
        return codelength(dist, pixel)

cpdef float mixture_of_gaussians(int var, np.ndarray[np.int_t, ndim=1] nbrs, int pixel, bint cl = False):
    cdef np.ndarray[np.float_t, ndim=1] dist = np.empty(256, np.float)
    cdef int i, j
    cdef int size = nbrs.shape[0]
    cdef float beta = - 0.5 / var
    for i in range(256):
        for j in range(size):
            dist[i] = exp(beta * (i - nbrs[j])**2)
    dist /= asum(dist)
    if not cl:
        return dist[pixel]
    else:
        return codelength(dist, pixel)


cpdef float mixture_of_laplacians(int var, np.ndarray[np.int_t, ndim=1] nbrs, int pixel, bint cl = False):
    cdef np.ndarray[np.float_t, ndim=1] dist = np.empty(256, np.float)
    cdef int i, j
    cdef int size = nbrs.shape[0]
    cdef float beta = - 1. / var
    for i in range(256):
        for j in range(size):
            dist[i] = exp(beta * abs(i - nbrs[j]))
    dist /= asum(dist)
    if not cl:
        return dist[pixel]
    else:
        return codelength(dist, pixel)



cpdef float uniform(np.ndarray[np.int_t] nbrs, int pixel, bint cl = False):
    if not cl:
        return 1. / 256
    else:
        return 8.


## get huffman code length from distribution
cpdef float codelength(np.ndarray[np.float_t] dist, int index):
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


cdef apply_func(np.ndarray[np.int_t, ndim=2] src, hierfunc func):
    cdef int length = src.shape[0]
    cdef np.ndarray[np.float_t, ndim=2] dest = np.zeros((length, length), dtype = float)
    cdef int level = length // 2
    cdef int i, j
    cdef int pixel
    cdef list nbr_indices
    cdef np.ndarray[np.int_t, ndim=1] nbrs
    while level > 0:
        print level
        for i from level <= i < length - 1 by 2 * level:
            for j from level <= j < length - 1 by 2 * level:
                nbr_indices = [(i - level, i - level, i + level, i + level),
                        (j - level, j + level, j - level, j + level)]
                pixel = src[i, j]
                nbrs = src[nbr_indices]
                dest[i, j] = func(nbrs, pixel)
        for i from 0 <= i < length - 1 by level:
            if i % (2 * level) == 0:
                for j from level <= j < length -1 by 2 * level:
                    if i == 0:
                        nbr_indices = [(i + level, i, i), (j, j - level, j + level)]
                    elif i == length - 1:
                        nbr_indices = [(i - level, i, i), (j, j - level, j + level)]
                    else:
                        nbr_indices = [(i, i, i - level, i + level), (j - level, j + level, j, j)]
                    pixel = src[i, j]
                    nbrs = src[nbr_indices]
                    dest[i, j] = func(nbrs, pixel)
            else:
                for j from 0 <= j < length - 1 by 2 * level:
                    if j == 0:
                        nbr_indices = [(i - level, i + level, i), (j, j, j + level)]
                    elif j == length - 1:
                        nbr_indices = [(i - level, i + level, i), (j, j, j - level)]
                    else:
                        nbr_indices = [(i, i, i - level, i + level), (j - level, j + level, j, j)]
                    pixel = src[i, j]
                    nbrs = src[nbr_indices]
                    dest[i, j] = func(nbrs, pixel)
        level = level / 2
    return dest



                


from PIL import Image
img = Image.open("../test-images/test-images-513/001-0.png")
src = np.asarray(img).astype(int)
