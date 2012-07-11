import numpy as np
from PIL import Image
from math import pi, cos, sin, sqrt

def make_patch(i, j, size = 64):
    theta = pi * i / size
    phi = 2 * pi * j / size
    a, b = cos(theta), sin(theta)
    c, d = cos(phi), sin(phi)
    P = lambda x, y: c * (a*x + b*y)**2 + d * (a*x + b*y)
    return np.array([[P(x, y) for y in [-1, 0, 1]] for x in [-1, 0, 1]])





def draw_patch(array, i_center, j_center, patch, scale = 5):
    ##scale should be odd
    M = np.max(patch)
    m = np.min(patch)
    a = 255.0 / (M - m)
    b = -a * m
    scale_colors = lambda x: a*x + b
    scaled_patch = np.kron(patch, np.ones((scale, scale)))
    scaled_patch = scale_colors(scaled_patch)
    shift = 3 * scale // 2
    array[i_center - shift: i_center + shift + 1, j_center - shift: j_center + shift + 1, 0] = scaled_patch
    array[i_center - shift: i_center + shift + 1, j_center - shift: j_center + shift + 1, 2] = 0
    




array = np.zeros((528, 528, 3), dtype = np.uint8)
array[:,:,2] = 255


p = make_patch(0, 0)

for i in range(10):
    for j in range(10):
        draw_patch(array, 528 - (8 + 56*i), (8 + 56*j), make_patch(7*i, 7*j))


img = Image.fromarray(array)
img.show()




img = Image.fromarray(array)
img.save("test.png")
