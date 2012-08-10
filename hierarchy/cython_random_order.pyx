#cython:profile=True



import numpy as np
cimport numpy as np
import random
from libc.math cimport sqrt, exp
from math import log
from numpy.linalg import lstsq


cpdef float dist(int x1, int y1, int x2, int y2):
    return abs(x1 - x2) + abs(y1 - y2)

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


def median(list values):
    sv = sorted(values)
    N = len(values)
    if N % 2 == 0:
        return (sv[N // 2] + sv[N // 2 - 1]) / 2
    else:
        return sv[N // 2]


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


def k_nearest_neighbors(int y, int x, data, mask, k):
    cdef int m = data.shape[0]
    cdef int n = data.shape[1]
    cdef int r = 1
    cdef int i, j, s
    cdef list nbrs = []
    cdef list weights = []
    cdef list indices = []
    while len(nbrs) < k:
        for s in range( 8 * r + 1):
            boundarypt(y, x, r, s, &i, &j)
            if len(nbrs) == k:
                break
            elif (0 <= i < m) and (0 <= j < n):
                if mask[i, j]:
                    nbrs.append(data[i, j])
                    weights.append(1. / dist(i, j, y, x))
                    indices.append((i - y, j - x))
        r += 1
    weights = map(lambda w: w / sum(weights), weights)
    return zip(nbrs, weights, indices)


def hierarchical_indices(int length):
    cdef list indices = []
    cdef int level = length // 2
    cdef int i, j, start
    while level > 0:
        for i from level <= i < length by 2 * level:
            for j from level <= j < length by 2 * level:
                indices.append((i, j))
        for i from 0 <= i < length by level:
            if i % (2 * level) == 0:
                start = level
            else:
                start = 0
            for j from start <= j < length by 2 * level:
                indices.append((i, j))
        level /= 2
    return indices


def mvl(nbrs, val):
    cdef float avg, var
    avg, var = meanvar(nbrs)
    cdef int nbr
    cdef float beta
    if var > 0:
        beta = - 1. / sqrt(var / 2)
    else:
        return (0.5, 0) if val == avg else (0.5 / 255, 0)
    cdef float total
    cdef float pred = 0
    cdef int i
    for nbr, weight, index in nbrs:
        total = 0
        for i in range(256):
            total += exp(beta * abs(i - nbr))
        pred += weight * exp(beta * abs(val - nbr)) / total
    return pred, var



def hyperplane(nbrs, val):
    M = np.matrix([[index[0], index[1], 1] for nbr, weight, index in nbrs])
    B = [nbr for nbr, weight, index in nbrs]
    a, b, c = lstsq(M, B)[0]
    return c ## TODO: replace by probability distribution


def edge(nbrs, val):
    M = np.matrix([[index[0], index[1], weight] for nbr, weight, index in nbrs])
    cdef list B = [nbr for nbr, weight, index in nbrs]
    a, b, c = lstsq(M, B)[0]
    ## sort wrt the gradient given by (a, b)
    sorted_nbrs = [(a * index[0] + b * index[1], nbr) for nbr, weight, index in nbrs]
    sorted_nbrs.sort()

    cdef int num_nbrs = len(sorted_nbrs)
    cdef float minerr = num_nbrs * 256**2
    cdef int E1, E2, i, thresh
    cdef float m1, m2
    res_avgs = (128, 128)
    thresh = 0
    for i in range(1, num_nbrs):
        E1 = 0
        E2 = 0
        m1 = 0
        m2 = 0
        ## compute the average and squared error on each of the two regions
        if i > 0:
            m1 = 1. * sum([v for d, v in sorted_nbrs[:i]]) / i
            ##m1 = median([v for d, v in sorted_nbrs[:i]])
            E1 = sum((x[1] - m1)**2 for x in sorted_nbrs[:i])
        if i < num_nbrs:
            m2 = sum([v for d, v in sorted_nbrs[i:]]) / (num_nbrs - i)
            ##m2 = median([v for d, v in sorted_nbrs[i:]])
            E2 = sum((x[1] - m2)**2 for x in sorted_nbrs[i:])
        if E1 + E2 < minerr:
            minerr = E1 + E2
            thresh = i
            res_avgs = (m1, m2)
    bdry = ((nbrs[thresh - 1][2][0] + nbrs[thresh][2][0]) / 2, (nbrs[thresh - 1][2][1] + nbrs[thresh][2][1])/2)
    if a * bdry[0] + b * bdry[1] < 0: 
        return res_avgs[1], minerr
    else:
        return res_avgs[0], minerr




def mvlmax(nbrs, val):
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
    cdef float maxprob = 0
    cdef int maxind = 0
    cdef float prob = 0
    for i in range(256):
        prob = 0
        for nbr, weight, index in nbrs:
            prob += weight * exp(beta * abs(i - nbr))
        if prob > maxprob:
            maxprob = prob
            maxind = i
    return maxind




def process(data, func = mvl, num_nbrs = 8, random_order = False, showerr = False):
    prediction = np.zeros(data.shape, dtype = float)
    errors = np.zeros(data.shape, dtype = float)
    seen_mask = np.zeros(data.shape, dtype = bool)
    five_percent = data.size // 20
    if random_order:
        indices = [(i, j) for i in range(data.shape[0]) for j in range(data.shape[1])]
        random.shuffle(indices)
    else:
        assert data.shape[0] == data.shape[1], "data array is not square"
        assert log(data.shape[0] - 1, 2) % 1 == 0, "data side length is not 2^k + 1"
        indices = hierarchical_indices(data.shape[0])
    count = num_nbrs
    for index in indices[:num_nbrs]:
        seen_mask[index] = True
        prediction[index] = 1. / 256
    for index in indices[num_nbrs:]:
        nbrs = k_nearest_neighbors(index[0], index[1], data, seen_mask, num_nbrs)
        if showerr:
            prediction[index], errors[index] = func(nbrs, data[index])
        else:
            prediction[index] = func(nbrs, data[index])
        seen_mask[index] = True
        count += 1
        if count % five_percent == 0:
            print 5 * count // five_percent
    if showerr:
        return prediction, errors
    else:
        return prediction



def logsum(arr):
    return -np.sum(np.log2(arr[arr > 0]))


def entropy(error_arr):
    L = np.array([np.sum((error_arr) % 256 == i) for i in range(256)], dtype = float)
    L /= np.sum(L)
    return np.sum(L[L > 0] * np.log2(L[L > 0]))
