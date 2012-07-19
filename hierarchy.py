from operator import mul
import pdb
from math import sqrt, pi, exp
import types
import numpy as np
from PIL import Image
from helper_routines import gaussian, combine_weights

def zipWith(func, A, B):
    return [func(a, b) for a, b in zip(A, B)]



def predict(pixel, nbrs, experts, weights):
    '''Given weights w_1, ..., w_k and experts P_1, ..., P_k, return sum(w_i * P_i[pixel | nbrs])
    as well as updated weights w'_i = w_i * P_i[pixel | nbrs].'''
    expert_probabilities = [exp(nbrs, pixel) for exp in experts]
    weighted_probabilities = (zipWith(mul, expert_probabilities, weights))
    aggregate_prob = sum(weighted_probabilities)
    new_local_weights = map(lambda w: w / aggregate_prob, weighted_probabilities)
    return aggregate_prob, new_local_weights





class Hierarchy:
    def __init__(self, source_array, experts): 
        '''source_array: two-dimensional numpy array of source image values.
        For now, source_array must be square with size (2^k + 1) x (2^k + 1).
        experts: list of functions [func1, func2, ...]. Each function must have the form func(nbrs, pixel)
        and cannot return 0.'''
        self.src = source_array.astype(int)
        self.experts = experts
        self.length, _ = self.src.shape
        self.probs = np.zeros((self.length, self.length))
        ## probs stores the probability assigned to each pixel by the hierarchical mixture-of-experts prediction
        self.weight_array = np.ones((self.length, self.length, len(experts)), dtype=float) / len(experts)
        ## weight_array associates an expert-weighting vector v to each pixel
        ## v = (v_0, v_1, ..., v_k) must satisfy \sum v_i = 1

    def process_pixel(self, index, nbr_indices):
        ## example: to get self.weight_array[0, 0], self.weight_array[1, 2], self.weight_array[3, 4], nbr_indices=[(0,1,3), (0, 2, 4)]
        pixel = self.src[index]
        num_nbrs = len(nbr_indices[0])
        weights = combine_weights(self.weight_array[nbr_indices])
        nbrs = self.src[nbr_indices]
        self.probs[index], self.weight_array[index] = predict(pixel, nbrs, self.experts, weights)
        
    def process_array(self):
        level = self.length // 2
        while level > 0:
            print level
            ##Assume that the Os below represent already processed pixels. 
            ##First process pixels in the locations labeled X:
            ##O   O   O
            ##  X   X
            ##O   O   O
            ##  X   X
            ##O   O   O
            for i in range(level, self.length, 2 * level):
                for j in range(level, self.length, 2 * level):
                    nbr_indices = [(i - level, i - level, i + level, i + level),
                            (j - level, j + level, j - level, j + level)]
                    self.process_pixel((i, j), nbr_indices)
            ## Now process pixels labeled Y:
            ##O Y O Y O
            ##Y X Y X Y
            ##O Y O Y O
            ##Y X Y X Y
            ##O Y O Y O
            ##
            for i in range(0, self.length, level):
                if i % (2 * level) == 0:
                    for j in range(level, self.length, 2 * level):
                        ## get the set of neighbors for pixels on the boundary of the image:
                        if i == 0:
                            nbr_indices = [(i + level, i, i), (j, j - level, j + level)]
                        elif i == self.length - 1:
                            nbr_indices = [(i - level, i, i), (j, j - level, j + level)]
                        else:
                            nbr_indices = [(i, i, i - level, i + level), (j - level, j + level, j, j)]
                        self.process_pixel((i, j), nbr_indices)
                else:
                    for j in range(0, self.length, 2 * level):
                        if j == 0:
                            nbr_indices = [(i - level, i + level, i), (j, j, j + level)]
                        elif j == self.length - 1:
                            nbr_indices = [(i - level, i + level, i), (j, j, j - level)]
                        else:
                            nbr_indices = [(i, i, i - level, i + level), (j - level, j + level, j, j)]
                        self.process_pixel((i, j), nbr_indices)
            level /= 2
    
    def entropy(self):
        return -np.sum(np.log2(self.probs[self.probs != 0]))

    def display_probs(self):
        m = np.max(self.probs)
        img = Image.fromarray((255 / m * self.probs).astype(np.uint8))
        img.show()
    
    def display_weights(self, expert_index):
        arr = self.weight_array[:,:,expert_index]
        m = np.max(self.weight_array)
        img = Image.fromarray((255 / m * arr).astype(np.uint8))
        img.show()






def uniform_expert(nbrs, pixel):
    return 1. / 256


