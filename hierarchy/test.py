from new_hierarchy import *
import os
from PIL import Image
from functools import partial
import numpy as np
import pickle

f = partial(mixture_of_laplacians, 20)

dest = open("results2.csv", 'w')
dest2 = open("pickled_results", 'w')
I = []
H = []
CL = []

for i in range(100):
    path = "../test-images/test-images-513/%03d-0.png" % (i + 1)
    img = Image.open(path)
    h = Hierarchy(img, [])
    cl = h.codelengths(f)
    I.append(img)
    CL.append(cl)
    H.append(h)
    o = os.path.getsize(path) * 8
    print i, np.sum(cl), '\t', o
    dest.write("%d\t%d\t%d\n" % (i, np.sum(cl), o))
    

pickle.dump([I, H, CL], dest2)
dest.close()
dest2.close()
    
