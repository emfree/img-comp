#cython:profile=True


import numpy as np
cimport numpy as np
import random
from libc.math cimport sqrt, exp



cdef tuple meanvar(list weighted_nbrs):
    cdef float mean = 0
    cdef float var = 0
    cdef int i
    cdef int size = len(weighted_nbrs)
    for i in range(size):
        mean += weighted_nbrs[i][1] * weighted_nbrs[i][0]
    for i in range(size):
        var += weighted_nbrs[i][1] * (weighted_nbrs[i][0] - mean)**2
    return mean, var


def mvl(nbrs, val):
    cdef float avg, var
    avg, var = meanvar(nbrs)
    cdef int nbr
    cdef float beta
    if var > 0:
        beta = - 1. / sqrt(var / 2)
    else:
        return 0.5 if val == avg else 0.5 / 255
    cdef float total
    cdef float pred = 0
    cdef int i
    for nbr, weight in nbrs:
        total = 0
        for i in range(256):
            total += exp(beta * abs(i - nbr))
        pred += weight * exp(beta * abs(val - nbr)) / total
    return  pred


cdef void boundarypt(int y, int x, int r, int s, int *i, int *j):
    cdef int d = s % (2 * r)
    if s <= 2 * r:
        i[0] = y - r
        j[0] = x - r + d
    elif 2 * r < s <= 4 * r:
        i[0] = y - r + d
        j[0] = x + r
    elif 4 * r < s <= 6 * r:
        i[0] = y + r
        j[0] = x + r - d
    elif 6 * r <= s:
        i[0] = y + r - d
        j[0] = x - r


cpdef float dist(int x1, int y1, int x2, int y2):
    return abs(x1 - x2) + abs(y1 - y2)


def k_nearest_neighbors(int y, int x, data, mask, k):
    cdef int m = data.shape[0]
    cdef int n = data.shape[1]
    cdef int r = 1
    cdef int i, j, s
    cdef list nbrs = []
    cdef list weights = []
    while len(nbrs) < k:
        for s in range( 8 * r + 1):
            boundarypt(y, x, r, s, &i, &j)
            if len(nbrs) == k:
                break
            elif (0 <= i < m) and (0 <= j < n):
                if mask[i, j]:
                    nbrs.append(data[i, j])
                    weights.append(1. / dist(i, j, y, x))
        r += 1
    weights = map(lambda w: w / sum(weights), weights)
    return zip(nbrs, weights)


def process_randomly(data, func = mvl, num_nbrs = 20):
    prediction = np.zeros(data.shape, dtype = float)
    seen_mask = np.zeros(data.shape, dtype = bool)
    indices = [(i, j) for i in range(data.shape[0]) for j in range(data.shape[1])]
    random.shuffle(indices)
    count = num_nbrs
    for index in indices[:num_nbrs]:
        seen_mask[index] = True
    for index in indices[num_nbrs:]:
        nbrs = k_nearest_neighbors(index[0], index[1], data, seen_mask, num_nbrs)
        prediction[index] = func(nbrs, data[index])
        seen_mask[index] = True
        count += 1
        if count % 1000 == 0:
            print count
    return prediction

        





