import numpy as np
from PIL import Image
import cython_random_order as cro
import random
from math import pi, sin, cos



def random_edge():
    edge = np.zeros((50, 50), dtype = int)
    angle = pi * random.random()
    print angle
    for i in range(50):
        for j in range(50):
            if (i - 25) * cos(angle) + (j - 25) * sin(angle) >= 0:
                edge[i, j] = 255
    ##Image.fromarray(edge.astype(np.uint8)).show()
    return edge
