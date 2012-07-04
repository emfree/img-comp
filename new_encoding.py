import numpy as np
from math import sqrt, pi, cos, sin, acos
from PIL import Image
import matplotlib.pyplot as plt
import e8
S = e8.Shell(8)

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
        klein_vects.append(normalize(vect))
        klein_patches.append(vect.reshape((3, 3))) ## todo: apply the appropriate transform
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



def project_whitened_array(array, quantized):
    m, n = array.shape
    projection = np.zeros((m / 3, n / 3))
    for i in range(0, m , 3):
        if i % 10 == 0:
            print i
        for j in range(0, n, 3):
            Y, X = slice(i, i + 3), slice(j, j + 3)
            r, s = i / 3, j / 3
            patch = array[Y, X].reshape(9)
            patch = np.dot(Transform_array, patch)
            projection[r, s] = np.argmin([np.dot(v, patch) for v in quantized])
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



##
img = Image.open("lena12.png")
array = np.asarray(img)[ : 510, : 510]
