import numpy as np
import random



def clip(i):
    if i < 0:
        return 0
    else:
        return i



def k_nearest_neighbors(index, data, mask, k = 4):
    i, j = index
    r = 0
    while np.sum(mask[clip(i - r) : i + r + 1, clip(j - r) : j + r + 1]) < k:
        r += 1
    Y = slice(clip(i - r), i + r + 1)
    X = slice(clip(j - r), j + r + 1)
    return data[Y, X][mask[Y, X]][:k]



def process_randomly(data, func, num_nbrs = 4):
    prediction = np.zeros(data.shape, dtype = float)
    seen_mask = np.zeros(data.shape, dtype = bool)
    indices = [(i, j) for i in range(data.shape[0]) for j in range(data.shape[1])]
    random.shuffle(indices)
    for index in indices[:num_nbrs]:
        seen_mask[index] = True
    for index in indices[num_nbrs:]:
        knn = k_nearest_neighbors(index, data, seen_mask, num_nbrs)
        prediction[index] = func(knn, data[index])
        seen_mask[index] = True
    return prediction

        





