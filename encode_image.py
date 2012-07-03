import numpy as np
import pickle
from math import sqrt, pi, cos, sin, acos, isnan
from PIL import Image
import huffman_code as hc



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
    klein_vects = np.zeros((size, size, 8))
    klein_patches = np.zeros((size, size, 3, 3))
    for i in range(size):
        ##print i
        theta = pi * i / size
        a, b = cos(theta), sin(theta)
        for j in range(size):
            phi = 2 * pi * j / size
            c, d = cos(phi), sin(phi)
            P = lambda x, y: c * (a*x + b*y)**2 + d * (a*x + b*y)
            vect = np.array([P(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]])
            klein_vects[i, j], mean, d_norm = normalize(vect)
            klein_patches[i,j] = (vect.reshape((3, 3)) - mean) / d_norm
    return klein_vects, klein_patches




sample_size = 16
Klein_vects, Klein_patches = make_klein_sample(sample_size)

f = open("test_error_distribution.txt")
p_list = pickle.load(f)
f.close()
huffman_dict, HuffmanTree = hc.huffman(p_list)






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


def byte_round(i):
    if i > 255:
        return 255
    if i < 0:
        return 0
    else:
        return int(i)

def encode_array(arr):
    m, n = arr.shape
    r, s = m//3, n//3 ## todo -- handle dimensions better
    output = np.zeros((r, s, 4), dtype = np.uint8)
    errors = np.zeros((m, n), dtype = np.int)
    d_norms = np.zeros((r, s) )
    round_patch = np.vectorize(byte_round)  ## todo -- make less of a hack
    for i in xrange(r):
        if i % 10 == 0:
            print i
        for j in xrange(s):
            y = slice(3 * i, 3 * (i + 1))  ## don't mix up x and y.
            x = slice(3 * j, 3 * (j + 1))
            patch = arr[y, x].flat
            vect, mean, d_norm = normalize(patch)
            d_norms[i, j] = d_norm
            mean = byte_round(mean)
            d_norm = byte_round(d_norm)
            if d_norm == 0:
                output[i, j, :] = [0, 0, mean, 0] ## so when decoding, first check whether the d_norm is zero or not.
                ## watch out -- d_norm could get ROUNDED to zero and then there could be errors
            else:
                proj_i, proj_j, max_dp = project_into_sample(vect, Klein_vects)
                ##print i, j
                output[i, j, :] = [proj_i, proj_j, mean, d_norm]
                errors[y, x] = arr[y, x].astype(int) - round_patch(mean + d_norm * Klein_patches[proj_i, proj_j])
    encoded_error = hc.encode(list(errors.flat), huffman_dict)
    return output, encoded_error, d_norms


def get_klein_patch(i, j, size = sample_size):
    theta = pi * i / size
    a, b = cos(theta), sin(theta)
    phi = 2 * pi * j / size
    c, d = cos(phi), sin(phi)
    P = lambda x, y: c * (a*x + b*y)**2 + d * (a*x + b*y)
    patch = np.matrix([P(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]])
    mean = np.sum(patch) / 9.0
    patch -= mean
    d_norm = sqrt(patch * D * patch.T)
    patch /= d_norm
    patch = np.array(patch).reshape((3, 3))
    return patch


def decode_array(arr, errors):
    ## lossy now
    r, s, d = arr.shape
    m = r * 3
    n = s * 3
    output = np.zeros((m, n))
    round_patch = np.vectorize(byte_round)
    ##error_array = np.array(hc.decode(errors, HuffmanTree), dtype = np.int).reshape((m, n))
    for i in xrange(r):
        ##print i
        for j in xrange(s):
            proj_i, proj_j, mean, d_norm = arr[i, j]
            d_norm = d_norm
            if d_norm == 0:
                output_patch = mean * np.ones((3, 3))
            else:
                src_patch = get_klein_patch(proj_i, proj_j)
                output_patch = round_patch((src_patch * d_norm) + mean)
            output[3*i:3*i+3, 3*j:3*j+3] = output_patch
    return (output).astype(np.uint8)



img = Image.open("lena12.png")
arr = np.asarray(img)[ : 510, : 510]
enc_array, enc_error, d_norms = encode_array(arr)
decoding = decode_array(enc_array, enc_error)
##img2 = Image.fromarray(decoding)
##img2.save("lena_test-5.png")
