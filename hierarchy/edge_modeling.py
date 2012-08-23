import numpy as np
from PIL import Image
import cython_random_order as cro
import random
from math import pi, sin, cos



def random_edge(size):
    edge = np.zeros((size, size), dtype = int)
    angle = pi * random.random()
    print angle
    for i in range(size):
        for j in range(size):
            if (i - size//2) * cos(angle) + (j - size//2) * sin(angle) >= 0:
                edge[i, j] = 255
    ##Image.fromarray(edge.astype(np.uint8)).show()
    return edge
