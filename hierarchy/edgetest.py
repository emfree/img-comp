import cython_random_order as cro
from PIL import Image
import numpy as np
import random



arr = np.zeros((40, 40), dtype = np.uint8)
for i in range(40):
    for j in range(40):
        arr[i, j] = 255 * (i+j) // 80

def process(data, func, num_nbrs = 8, random_order = False, draw = False):
    prediction = np.zeros(data.shape, dtype = float)
    ##errors = np.zeros(data.shape, dtype = float)
    seen_mask = np.zeros(data.shape, dtype = bool)
    five_percent = data.size // 20
    if random_order:
        indices = [(i, j) for i in range(data.shape[0]) for j in range(data.shape[1])]
        random.shuffle(indices)
    else:
        assert data.shape[0] == data.shape[1], "data array is not square"
        assert log(data.shape[0] - 1, 2) % 1 == 0, "data side length is not 2^k + 1"
        indices = hierarchical_indices(data.shape[0])
    count = num_nbrs
    if draw:
        img = np.zeros((data.shape[0], data.shape[1], 3), dtype = np.uint8)
    for index in indices[:num_nbrs]:
        seen_mask[index] = True
        prediction[index] = 1. / 256
        if draw:
            img[index] = [255 - data[index], 0, data[index]]
    for index in indices[num_nbrs:]:
        nbrs = cro.k_nearest_neighbors(index[0], index[1], data, seen_mask, num_nbrs)
        ##prediction[index], errors[index] = func(nbrs, data[index])
        prediction[index] = func(nbrs, data[index])
        if draw:
            img[index[0], index[1], 0] = 255 - prediction[index]
            img[index[0], index[1], 2] = prediction[index]
            Image.fromarray(img).save("./tmp_images/%d.png" % count)
        seen_mask[index] = True
        count += 1
        if count % five_percent == 0:
            print 5 * count // five_percent
    return prediction

##P = process(arr, cro.edge, random_order = True, draw = True)
