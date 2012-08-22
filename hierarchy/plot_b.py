from hierarchy2 import *
from PIL import Image
from numpy import *
import cython_random_order as cro

def path(n): return '../test-images/test-images-513/%03d-0.png' % n


def getimg(n): return Image.open(path(n))


def data(img):
    pred, metadata = predict(img)
    B = array([sum(abs(n - average(nbrs)) for n in nbrs) * 1. / len(nbrs) for p, i, j, v, nbrs in metadata])
    M = array([min(abs(v - n) for n in nbrs) for p, i, j, v, nbrs in metadata])
    return B, M



def bpredictor(B, M):
    return lambda v: average(M[B == v])


def entropy(arr):
    L = zeros(256)
    for i in range(256):
        L[i] = sum((arr % 256) == i)
    L = L * 1. / sum(L)
    return sum(L[L > 0] * log2(L[L > 0]))



img = asarray(getimg(1), dtype = int)[-129:, -129:]
