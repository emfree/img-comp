import pstats, cProfile
from PIL import Image
import numpy as np

import pyximport
pyximport.install()

import cython_random_order as cro


path = '../test-images/test-images-513/001-0.png'
img = Image.open(path)
data = np.asarray(img, dtype = int)[-129:, -129:]


cProfile.runctx("cro.process(data, num_nbrs = 8)", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
