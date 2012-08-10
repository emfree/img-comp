import pstats, cProfile
from PIL import Image
import numpy as np

import pyximport
pyximport.install()

import cython_random_order as cro

def test():
    img = Image.open('../test-images/test-images-513/001-0.png')
    arr = np.asarray(img, dtype = int)
    pred = cro.process(arr, cro.edge, num_nbrs = 4)


cProfile.runctx("test()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

