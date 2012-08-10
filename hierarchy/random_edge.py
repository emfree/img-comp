from random import shuffle, randrange
import numpy as np
import cython_random_order as cro
from PIL import Image
from numpy.linalg import lstsq



def edge(nbrs, val):
    M = np.matrix([[index[0], index[1], 1] for nbr, weight, index in nbrs])
    B = [nbr for nbr, weight, index in nbrs]
    a, b, c = lstsq(M, B)[0]
    print a, b
    ## sort wrt the gradient given by (a, b)
    sorted_nbrs = [(a * index[0] + b * index[1], nbr) for nbr, weight, index in nbrs]
    sorted_nbrs.sort()
    num_nbrs = len(sorted_nbrs)
    minerr = num_nbrs * 256**2
    avgs = []
    thresh = 0
    #import pdb; pdb.set_trace()
    for i in range(num_nbrs + 1):
        E1 = 0
        E2 = 0
        m1 = 0
        m2 = 0
        ## compute the average and squared error on each of the two regions
        if i > 0:
            m1 = 1. * sum([v for d, v in sorted_nbrs[:i]]) / i
            E1 = sum((x[1] - m1)**2 for x in sorted_nbrs[:i])
        if i < num_nbrs:
            m2 = 1. * sum([v for d, v in sorted_nbrs[i:]]) / (num_nbrs - i)
            E2 = sum((x[1] - m2)**2 for x in sorted_nbrs[i:])
        avgs.append((m1, m2))
        if E1 + E2 < minerr:
            minerr = E1 + E2
            thresh = i
    if thresh == 0:
        return avgs[0][1]
    elif thresh == num_nbrs:
        return avgs[-1][0]
    else:
        bdry = ((nbrs[thresh - 1][2][0] + nbrs[thresh][2][0]) / 2, (nbrs[thresh - 1][2][1] + nbrs[thresh][2][1]) / 2)
        if a * bdry[0] + b * bdry[1] < 0: 
            return avgs[thresh][1]
        else:
            return avgs[thresh][0]



def random_image():
    indices = [(i, j) for i in range(0, 20) for j in range(0, 20)]
    shuffle(indices)
    random_image = np.zeros((20, 20, 3), dtype = np.uint8)
    seen_mask = np.zeros((20, 20), dtype = bool)
    ## specific initialization
    indices.remove((0,0))
    indices.remove((0,1))
    indices.remove((19,19))
    indices.remove((19, 18))
    indices.insert(0, (0, 0))
    indices.insert(0, (0, 1))
    indices.insert(0, (19, 19))
    indices.insert(0, (19, 18))
    random_image[0,0] = 0
    random_image[0,1] = 0
    random_image[19, 19] = 255
    random_image[19, 18] = 255
    seen_mask[0,0] = seen_mask[0,1] = seen_mask[19,19] = seen_mask[19,18] = True
    ##for index in indices[:4]:
    ##    random_image[index] = randrange(0, 256)
    ##    seen_mask[index] = True
    count = 4
    for index in indices[4:]:
        Image.fromarray(random_image).save("./tmp_images/%d.png" % count)
        nbrs = cro.k_nearest_neighbors(index[0], index[1], random_image, seen_mask, count)
        random_image[index] = cro.edge(nbrs, random_image[index])
        print index, random_image[index]
        seen_mask[index] = True
        count += 1
    return random_image

