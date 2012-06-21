import numpy as np
from math import sqrt, pi, cos, sin, acos, isnan
from PIL import Image



def euclid_norm(list):
    return sqrt(sum([x**2 for x in list]))





def arc_dist(vect1, vect2):
    dp = np.dot(vect1, vect2)
    if dp > 1:
        return 0
    elif dp < -1:
        return pi
    else:
        return acos(dp) 


D = np.matrix([[2, -1, 0, -1, 0, 0, 0, 0, 0],
     [-1, 3, -1, 0, -1, 0, 0, 0, 0],
     [0, -1, 2, 0, 0, -1, 0, 0, 0],
     [-1, 0, 0, 3, -1, 0, -1, 0, 0],
     [0, -1, 0, -1, 4, -1, 0, -1, 0],
     [0, 0, -1, 0, -1, 3, 0, 0, -1],
     [0, 0, 0, -1, 0, 0, 2, -1, 0],
     [0, 0, 0, 0, -1, 0, -1, 3, -1],
     [0, 0, 0, 0, 0, -1, 0, -1, 2]])


DCT_basis = np.matrix( [1/sqrt(6) * np.array([1, 0, -1, 1, 0, -1, 1, 0, -1]), 
               1/sqrt(6) * np.array([1, 1, 1, 0, 0, 0, -1, -1, -1]),
               1/sqrt(54) * np.array([1, -2, 1, 1, -2, 1, 1, -2, 1]),
               1/sqrt(54) * np.array([1, 1, 1, -2, -2, -2, 1, 1, 1]),
               1/sqrt(8) * np.array([1, 0, -1, 0, 0, 0, -1, 0, 1]),
               1/sqrt(48) * np.array([1, 0, -1, -2, 0, 2, 1, 0, -1]),
               1/sqrt(48) * np.array([1, -2, 1, 0, 0, 0, -1, 2, -1]),
               1/sqrt(216) * np.array([1, -2, 1, -2, 4, -2, 1, -2, 1]) ]).T

Lambda = np.diag([1 / float(v * v.T) for v in DCT_basis.T])

Transform_matrix = Lambda * DCT_basis.T


def normalize(patch):
    '''Takes as argument a 9-element list (or 1D numpy array). Returns a tuple consisting of the corresponding D-normalized vector as an 8-element list of coefficients (wrt the DCT basis), the mean, and the D-norm of the patch.'''
    mean = np.sum(patch) / 9.0
    output = np.matrix(patch - mean).T  ## subtract the mean
    d_norm = sqrt( output.T*D*output )
    output /= d_norm ## output now has D-norm 1
    output = Transform_matrix * output ## now it's in the DCT basis, which means it's an 8-vector of Euclidean norm 1
    output = list(output.flat) ## turn it into a list
    return output, mean, d_norm
    


def make_klein_sample(size):
    klein_sample = np.zeros((size, size, 8))
    for i in range(size):
        ##print i
        theta = pi * i / size
        a, b = cos(theta), sin(theta)
        for j in range(size):
            phi = 2 * pi * j / size
            c, d = cos(phi), sin(phi)
            P = lambda x, y: c * (a*x + b*y)**2 + d * (a*x + b*y)
            vect = np.array([P(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]])
            klein_sample[i, j] = normalize(vect)[0]
    return klein_sample


sample_size = 64 
Klein_sample = make_klein_sample(sample_size)


def project_into_sample(vect, sample_array):
    m, n, d = sample_array.shape
    max_dp = float("-infinity")
    for i in range(m):
        for j in range(n):
            new_dp = np.dot(vect, sample_array[i, j])
            if new_dp > max_dp:
                best_i, best_j = i, j
                max_dp = new_dp
    return best_i, best_j, max_dp


def transform_array(arr):
    m, n = arr.shape
    vects = np.zeros((m, n, 8))
    d_norms = np.zeros((m, n))
    for i in range(1, m-1):
        print i
        for j in range(1, n-1):
            patch = list(arr[i-1:i+2, j-1:j+2].flat)
            ##patch = np.log(patch)
            vect, mean, d_norm = normalize(patch)
            vects[i, j, :] = vect
            d_norms[i, j] = d_norm
    return vects, d_norms


def threshold(vects, d_norms, thresh):
    m, n = d_norms.shape
    return [vects[i, j] for i in range(m) for j in range(n) if d_norms[i, j] > thresh]

def array_stats(data):
    output = []
    K = sample_size * [sample_size * [[]]]
    data_size = len(data)
    incr = data_size / 20
    for i in xrange(data_size):
        if i % incr == 0:
            print "%d percent done" % (5 * i / incr)
        i_coord, j_coord, min_dist = project_into_sample(data[i], Klein_sample)
        output.append(min_dist)
        K[i_coord][j_coord].append(data[i]) ## now K is an array of lists.
        ## Each list contains the S^7 vectors that got mapped to the corresponding Klein bottle point.
        ## len(K[i][j] recovers the number of patches that got mapped to the pt.
    return output, K


img = Image.open("lena12.png")
arr = np.asarray(img)




