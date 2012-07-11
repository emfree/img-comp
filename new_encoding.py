import numpy as np
from math import sqrt, pi, cos, sin, acos, log
from PIL import Image
import matplotlib.pyplot as plt
import e8

## for simplicity, assume array dimensions are divisible by 3
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
Transform_array = np.array(Transform_matrix)



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



def whiten_array(array):
    m, n = array.shape
    means = np.copy(array).astype(float)
    for i in range(0, m, 3):
        for j in range(0, n, 3):
            Y, X = slice(i, i + 3), slice(j, j + 3)
            means[Y, X] = np.average(array[Y, X])
    whitened_array = array.astype(float) - means
    return whitened_array, means


def contrasts(whitened_array):
    m, n = whitened_array.shape
    contrasts = np.copy(whitened_array)
    for i in range(0, m, 3):
        for j in range(0, n, 3):
            Y, X = slice(i, i + 3), slice(j, j + 3)
            contrast = np.sum(whitened_array[Y, X]**2)
            contrasts[Y, X] = contrast
    return contrasts
            


def project_whitened_array(array, quantized):
    m, n = array.shape
    projection = np.zeros((m / 3, n / 3), dtype = np.int)
    for i in range(0, m , 3):
        if i % 10 == 0:
            print i
        for j in range(0, n, 3):
            Y, X = slice(i, i + 3), slice(j, j + 3)
            r, s = i / 3, j / 3
            patch = array[Y, X].reshape(9)
            patch = np.dot(Transform_array, patch)
            projection[r, s] = np.argmax([np.dot(v, patch) for v in quantized])
    return projection


def project_into_e8(array, S):
    m, n = array.shape
    projection = []
    for i in range(0, m , 3):
        if i % 10 == 0:
            print i
        for j in range(0, n, 3):
            Y, X = slice(i, i + 3), slice(j, j + 3)
            r, s = i / 3, j / 3
            patch = array[Y, X].reshape(9)
            patch = np.dot(Transform_array, patch)
            projection.append(S.project(list(patch)))
    return projection


def encode(array, KV, KP):
    W, M = whiten_array(array)
    P = project_whitened_array(W, KV)
    S = np.zeros((510, 510))
    for i in range(0, 510, 3):
        for j in range(0, 510, 3):
            src = W[i:i+3, j:j+3].flat
            proj = KP[P[i/3, j/3]].flat
            scale = np.dot(src, proj)
            S[i:i+3, j:j+3] = scale * np.ones((3, 3))


    S = S.astype(int)
    M = M.astype(int)
    return M, P, S


def decode(M, P, S, KP):
    RW = np.zeros((510, 510))
    for i in range(0, 510, 3):
        for j in range(0, 510, 3):
            RW[i:i+3, j:j+3] = S[i, j] * KP[P[i/3, j/3]]

    RI = (RW + M).astype(int)
    return RI


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


def compare_errors(error1, error2):
    m, n = error1.shape
    result = np.zeros((m, n), dtype = bool)
    for i in range(0, m, 3):
        for j in range(0, n, 3):
            Y, X = slice(i, i + 3), slice(j, j + 3)
            if np.sum(np.abs(error1[Y, X])) > np.sum(np.abs(error2[Y, X])):
                result[Y, X] = np.ones((3, 3), dtype = bool)
    return result


img = Image.open("lena12.png")
array = np.asarray(img)[ : 510, : 510].astype(np.float)
KV, KP = make_klein_sample([(i * pi / 16, j * 2 * pi / 16) for i in range(16) for j in range(16)])
#M, P, S = encode(array, KV, KP)
#decoding = decode(M, P, S, KP)

##S = e8.Shell(8)
