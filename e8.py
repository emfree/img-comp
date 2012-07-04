import itertools
import numpy as np
from math import sqrt





def get_norm_squared(v):
    return sum([x**2 for x in v])


def shift(list, offset):
    return [x + offset for x in list]

def dot(v1, v2):
    return sum(x1 * x2 for x1, x2 in zip(v1, v2))


def get_sort_map(list):
    length = len(list)
    pairs = sorted((list[i], i) for i in range(length))
    sorted_list, sort_map = zip(*pairs)
    return sorted_list, sort_map

## sort_map[i] = index in list of the i-th smallest element of list.



def remap(sorted_list, sort_map, N):
    original_list = N * [0]
    for n in range(N):
        original_list[sort_map[n]] = sorted_list[n]
    return original_list


class Shell:
    def __init__(self, square_norm):
        N = int(sqrt(square_norm))
        integral_vects = (v for v in itertools.product(range( -N, N + 1), repeat = 8) if get_norm_squared(v) == square_norm and sum(v) % 2 == 0)
        half_integral_vects = (v for v in itertools.product(shift(range( -N - 1, N + 1), 0.5), repeat = 8) 
            if get_norm_squared(v) == square_norm and sum(v) % 2 == 0)
        self.shell = sorted(itertools.chain(integral_vects, half_integral_vects))
        self.shell_reduced = list(set([tuple(sorted(v)) for v in self.shell]))
        self.reduced_size = len(self.shell_reduced)
        self.coordinate_values = range( -N, N + 1) + shift(range( -N - 1, N + 1), 0.5)

    def project(self, vector):
        sorted_vector, sort_map = get_sort_map(vector)
        min_dotprod, projection_index = min((dot(vector, self.shell_reduced[index]), index) for index in range(self.reduced_size))
        sorted_projection = self.shell_reduced[projection_index]
        return remap(sorted_projection, sort_map, 8)
