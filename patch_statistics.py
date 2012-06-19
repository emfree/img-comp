import numpy as np
from math import sqrt, pi, cos, sin, acos
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
        print i
        theta = i / (2*pi)
        a, b = cos(theta), sin(theta)
        for j in range(size):
            phi = j / (2*pi)
            c, d = cos(phi), sin(phi)
            P = lambda x, y: a * (c*x + d*y)**2 + b * (c*x + d*y)
            vect = np.array([P(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]])
            klein_sample[i, j] = normalize(vect)[0]
    return klein_sample


sample_size = 64 
Klein_sample = make_klein_sample(sample_size)


def project_into_sample(vect, sample_array):
    m, n, d = sample_array.shape
    min_dist = float("infinity")
    for i in range(m):
        for j in range(n):
            new_dist = arc_dist(vect, sample_array[i, j])
            if new_dist < min_dist:
                min_i, min_j = i, j
                min_dist = new_dist
    return min_i, min_j, min_dist


def threshold_array(arr, thresh):
    m, n = arr.shape
    output = []
    for i in range(1, m-1):
        print i
        for j in range(1, n-1):
            patch = list(arr[i-1:i+2, j-1:j+2].flat)
            patch = np.log(patch)
            vect, mean, d_norm = normalize(patch)
            if d_norm > thresh:
                output.append(vect)
    return output



def array_stats(sample):
    output = [project_into_sample(vect, Klein_sample) for vect in sample]
    return output


img = Image.open("lena12.png")
arr = np.asarray(img)

def pts_in_nbhd(src_pt, sample, max_dist):
    output = []
    for pt in sample:
        d = arc_dist(pt, src_pt)
        if d < max_dist:
            output.append((pt, d))
    return output


def getdata(sample, max_dist):
    data = np.zeros((32, 32))
    for i in range(32):
        print i
        for j in range(32):
            data[i, j] = len(pts_in_nbhd(Klein_sample[i,j,:], sample, max_dist))
    return data

