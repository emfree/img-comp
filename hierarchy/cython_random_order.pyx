#cython:profile=True


import bisect
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp
from libcpp.vector cimport vector
from math import log
from numpy.linalg import lstsq


cdef: 
    struct NeighborData:
        int value
        float weight
        int y
        int x


cpdef float dist(int x1, int y1, int x2, int y2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

cdef tuple meanvar(vector[NeighborData] weighted_nbrs):
    cdef float sum = 0
    cdef float sum_of_squares = 0
    for nbr in weighted_nbrs:
        sum += nbr.weight * nbr.value
        sum_of_squares += nbr.weight * nbr.value**2
    mean = sum / weighted_nbrs.size()
    return mean, sum_of_squares / weighted_nbrs.size() - mean**2


## todo: there should be a less ugly way to do this
cpdef tuple pymeanvar(list index_value_tuples):
    cdef float sum = 0
    cdef float sum_of_squares = 0
    for tuple in index_value_tuples:
        sum += tuple[1]
        sum_of_squares += tuple[1]**2
    mean = 1. * sum / len(index_value_tuples)
    return mean, 1. * sum_of_squares / len(index_value_tuples) - mean**2






def median(list values):
    sv = sorted(values)
    N = len(values)
    if N % 2 == 0:
        return (sv[N // 2] + sv[N // 2 - 1]) / 2
    else:
        return sv[N // 2]


cdef void boundarypt(int y, int x, int r, int s, int *i, int *j):
    cdef int d = s % (2 * r)
    if s < 2 * r:
        i[0] = y - r
        j[0] = x - r + d
    elif 2 * r <= s < 4 * r:
        i[0] = y - r + d
        j[0] = x + r
    elif 4 * r <= s < 6 * r:
        i[0] = y + r
        j[0] = x + r - d
    elif 6 * r <= s:
        i[0] = y + r - d
        j[0] = x - r


def pyboundarypt(y, x, r, s):
    cdef int i, j
    boundarypt(y, x, r, s, &i, &j)
    return i, j


cdef vector[NeighborData] nearest_neighbors(index, data, ordering, k):
    y, x = index
    cdef vector[NeighborData] nbrs
    cdef int m = data.shape[0]
    cdef int n = data.shape[1]
    cdef int r = 1
    cdef int i, j, s
    cdef float weight
    cdef float weightsum = 0
    cdef NeighborData nbr
    while nbrs.size() < k:
        for s in range(8 * r):
            boundarypt(y, x, r, s, &i, &j)
            if nbrs.size() == k:
                break
            elif (0 <= i < m) and (0 <= j < n):
                if ordering[i, j] < ordering[y, x]:
                    weight = exp(-dist(i, j, y, x))
                    weightsum += weight
                    nbr = NeighborData(data[i, j], weight, i - y, j - x)
                    nbrs.push_back(nbr)
        r += 1
    for i in range(nbrs.size()):
        nbrs[i].weight /= weightsum
    return nbrs


##def hierarchical_indices(int length):
##    cdef list indices = [(0, 0), (0, length - 1), (length - 1, 0), (length - 1, length - 1)]
##    cdef int level = length // 2
##    cdef int i, j, start
##    while level > 0:
##        for i from level <= i < length by 2 * level:
##            for j from level <= j < length by 2 * level:
##                indices.append((i, j))
##        for i from 0 <= i < length by level:
##            if i % (2 * level) == 0:
##                start = level
##            else:
##                start = 0
##            for j from start <= j < length by 2 * level:
##                indices.append((i, j))
##        level /= 2
##    return indices



def laplace(float mu, float beta, int x):
    cdef float total = .5 * (2 - exp(abs(255.5 - mu) * beta) - exp(abs(mu - .5) * beta))
    if x == mu:
        return (1 - exp(beta / 2.)) / total
    else:
        return .5 * ( exp(beta * (-.5 + abs(mu - x))) - exp(beta * (.5 + abs(mu - x)))) / total



cdef float mvl(vector[NeighborData] nbrs, int val):
    cdef float avg, var
    avg, var = meanvar(nbrs)
    cdef float beta
    if var > 2:
        beta = - 1. / sqrt(var / 2)
    else:
        beta = - 1.
    cdef float pred = 0
    for nbr in nbrs:
        pred += nbr.weight * laplace(nbr.value, beta, val)
    return pred



##TODO: replace nbrs by vector type


cdef float hyperplane(vector[NeighborData] nbrs, int val):
    M = np.zeros((nbrs.size(), 3))
    B = np.zeros(nbrs.size())
    cdef int i
    for i in range(nbrs.size()):
        M[i] = [nbrs[i].y, nbrs[i].x, 1]
        B[i] = nbrs[i].value
    cdef float a, b, c
    a, b, c = lstsq(M, B)[0]
    return c ## TODO: replace by probability distribution






cdef float edge(vector[NeighborData] nbrs, int val):
    M = np.zeros((nbrs.size(), 3))
    B = np.zeros(nbrs.size())
    cdef int i, thresh
    for i in range(nbrs.size()):
        M[i] = [nbrs[i].y, nbrs[i].x, 1]
        B[i] = nbrs[i].value
    cdef float a, b, c
    a, b, c = lstsq(M, B)[0]
    ## list the neighbors sorted wrt the gradient given by (a, b)
    sorted_nbrs = []
    for nbr in nbrs:
        bisect.insort(sorted_nbrs, (a * nbr.y + b * nbr.x, nbr.value, nbr.y, nbr.x))
    cdef int num_nbrs = len(sorted_nbrs)
    cdef float minerr = num_nbrs * 256**2
    cdef float mean1, mean2, E1, E2
    res_avgs = (128, 128)
    thresh = 0
    for i in range(1, num_nbrs - 1):
        ## compute the average and squared error on each of the two regions
        mean1, E1 = pymeanvar(sorted_nbrs[:i])
        mean2, E2 = pymeanvar(sorted_nbrs[i:])
        if E1 + E2 < minerr:
            minerr = E1 + E2
            thresh = i
            res_avgs = (mean1, mean2)
    bdry = ((sorted_nbrs[thresh - 1][2] + sorted_nbrs[thresh][2]) / 2, (sorted_nbrs[thresh - 1][3] + sorted_nbrs[thresh][3])/2)
    if a * bdry[0] + b * bdry[1] < 0: 
        return res_avgs[1]
    else:
        return res_avgs[0] ## TODO: replace by probability distribution



def process(data, num_nbrs = 8, num_seeds = 8, expert = 'mvl'):
    assert num_seeds >= num_nbrs, "must have at least as many seeds as neighbors"
    cdef float (*func)(vector[NeighborData], int)
    if expert == 'mvl':
        func = &mvl
    else:
        func = &edge
    prediction = np.zeros(data.shape, dtype = float)
    ## define a random ordering of the array locations in data
    ordering = np.random.permutation(data.size).reshape(data.shape)
    print ordering
    ## create an iterator over data
    it = np.nditer(data, flags = ['multi_index'])
    cdef vector[NeighborData] nbrs
    while not it.finished:
        if ordering[it.multi_index] < num_seeds:
            ## just use the uniform distribution for the first num_seeds pixels
            prediction[it.multi_index] = 1. / 256
        else:
            nbrs = nearest_neighbors(it.multi_index, data, ordering, num_nbrs)
            prediction[it.multi_index] = func(nbrs, it[0])
        it.iternext()
    return prediction




##def random_reconstruct(data, num_nbrs, num_seeds, random_order = False, sigma = 1.):
##    from scipy.stats import rv_discrete
##    reconstruction = np.zeros(data.shape, dtype = int)
##    seen_mask = np.zeros(data.shape, dtype = bool)
##    five_percent = data.size // 20
##    if random_order:
##        indices = [(i, j) for i in range(data.shape[0]) for j in range(data.shape[1])]
##        random.shuffle(indices)
##    else:
##        assert data.shape[0] == data.shape[1], "data array is not square"
##        assert log(data.shape[0] - 1, 2) % 1 == 0, "data side length is not 2^k + 1"
##        indices = hierarchical_indices(data.shape[0])
##    count = num_nbrs
##    for index in indices[:num_seeds]:
##        seen_mask[index] = True
##        reconstruction[index] = data[index]
##    for index in indices[num_seeds:]:
##        nbrs = k_nearest_neighbors(index[0], index[1], reconstruction, seen_mask, num_nbrs, sigma)
##        distro = rv_discrete(0, 255, name = 'mvl', values = (range(256), [mvl2(nbrs, i) for i in range(256)]))
##        reconstruction[index] = distro.rvs()
##        seen_mask[index] = True
##        count += 1
##        if count % five_percent == 0:
##            print 5 * count // five_percent
##    return reconstruction




def logsum(arr):
    return -np.sum(np.log2(arr[arr > 0]))


def entropy(error_arr):
    L = np.array([np.sum((error_arr) % 256 == i) for i in range(256)], dtype = float)
    L /= np.sum(L)
    return np.sum(L[L > 0] * np.log2(L[L > 0]))
