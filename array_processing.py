

from numpy.lib.stride_tricks import as_strided
import numpy as np
from math import sqrt, pi, cos, sin, log
from PIL import Image
import matplotlib.pyplot as plt

##for now
##quantized: one patch / vector = one row

def frequency(arr):
    ## assume arr is integer-valued
    m = np.min(arr)
    M = np.max(arr)
    freq = []
    for i in range(m, M + 1):
        freq.append(np.sum(arr == i))
    return freq

def entropy(dist):
    total = np.sum(dist)
    return -sum(1. * x / total * log(1. * x / total, 2) for x in dist if x > 0)



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
transformer = np.array(Lambda * DCT_basis.T)

def normalize(patch):
    mean = np.sum(patch) / 9.0
    output = np.matrix(patch - mean).T  ## subtract the mean
    d_norm = sqrt( output.T*D*output )
    output /= d_norm ## output now has D-norm 1
    output = Transform_matrix * output ## now it's in the DCT basis, which means it's an 8-vector of Euclidean norm 1
    output = list(output.flat) ## turn it into a list
    return output
    


def make_klein_sample(parameters):
    N = len(parameters)
    klein_vects = []
    klein_patches = []
    for theta, phi in parameters:
        a, b = cos(theta), sin(theta)
        c, d = cos(phi), sin(phi)
        P = lambda x, y: c * (a*x + b*y)**2 + d * (a*x + b*y)
        vect = np.array([P(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]])
        vect -= np.average(vect)
        vect /= sqrt(np.sum(vect**2))
        klein_vects.append(normalize(vect))
        klein_patches.append(vect.reshape((3, 3))) ## patches are assumed to have Euclidean norm 1
    return klein_vects, klein_patches


KV, KP = make_klein_sample([(i * pi / 16, j * 2 * pi / 16) for i in range(16) for j in range(16)])
KV = np.array(KV)
KP = np.array(KP).reshape((256, 9))


def preprocess(array):
    m, n = array.shape
    assert m % 3 == 0 and n % 3 == 0, "array dimensions are not multiples of 3"
    long_stride, short_stride = array.strides
    reshaped_array = as_strided(array, (m /3, n/3, 3, 3), (3*long_stride, 3*short_stride, long_stride, short_stride)).reshape((m * n / 9, 9)).T
    return reshaped_array



def whiten(p_array):
    means = np.apply_along_axis(lambda x: sum(x) / 9.0, 0, p_array)
    whitened = (p_array - means)
    return means, whitened



def project(p_array, quantized):
    t_array = np.dot(transformer, p_array)
    results = np.dot(quantized, t_array)
    return np.apply_along_axis(np.argmax, 0, results)
    


def encode(array, sphere_quantized, patch_quantized): 
    means, whitened = whiten(preprocess(array))
    projection = project(whitened, sphere_quantized)
    scales = np.apply_along_axis(sum, 0, patch_quantized[projection].T * whitened)
    return means, projection, scales



def decode(means, projection, scales, patch_quantized):
    array = patch_quantized[projection]
    return (array.T * scales + means).T 
    

def deprocess(decoding, shape):
    m, n = shape
    return decoding.reshape((m, n, 3, 3)).swapaxes(1, 2).reshape((3*m, 3*n))


array = np.asarray(Image.open("lena12.png")).astype(np.int)[:510, :510]

