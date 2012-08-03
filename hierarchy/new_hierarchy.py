from PIL import Image
import numpy as np
from math import log, exp, sqrt, pi
from aux import order2
import matplotlib.pyplot as plt
from experts import *
from functools import partial
from collections import defaultdict


defaultlog = np.vectorize(lambda x: log(x, 2) if x > 0 else 0)



class Hierarchy:
    def __init__(self, data, experts):
        ## experts should be a dictionary of key: function pairs
        self.src = np.asarray(data, dtype = int)
        self.length, n = self.src.shape
        self.experts = []
        self.predictions = []
        assert self.length == n, "shape of data is not square"
        assert log(self.length-1, 2) % 1 == 0, "side length of data does not have form 2^k + 1"
        self.apply_experts(experts)

    def apply_func(self, src, func, dest):
        level = self.length // 2
        while level > 0:
            print level
            for i in range(level, self.length, 2 * level):
                for j in range(level, self.length, 2 * level):
                    nbr_indices = [(i - level, i - level, i + level, i + level),
                            (j - level, j + level, j - level, j + level)]
                    pixel = src[i, j]
                    nbrs = src[nbr_indices]
                    dest[i, j] = func(nbrs, pixel)
            for i in range(0, self.length, level):
                if i % (2 * level) == 0:
                    for j in range(level, self.length, 2 * level):
                        if i == 0:
                            nbr_indices = [(i + level, i, i), (j, j - level, j + level)]
                        elif i == self.length - 1:
                            nbr_indices = [(i - level, i, i), (j, j - level, j + level)]
                        else:
                            nbr_indices = [(i, i, i - level, i + level), (j - level, j + level, j, j)]
                        pixel = src[i, j]
                        nbrs = src[nbr_indices]
                        dest[i, j] = func(nbrs, pixel)
                else:
                    for j in range(0, self.length, 2 * level):
                        if j == 0:
                            nbr_indices = [(i - level, i + level, i), (j, j, j + level)]
                        elif j == self.length - 1:
                            nbr_indices = [(i - level, i + level, i), (j, j, j - level)]
                        else:
                            nbr_indices = [(i, i, i - level, i + level), (j - level, j + level, j, j)]
                        pixel = src[i, j]
                        nbrs = src[nbr_indices]
                        dest[i, j] = func(nbrs, pixel)
            level /= 2

    def apply_experts(self, experts):
        for exp in experts:
            self.experts.append(exp)
            self.predictions.append(np.zeros((self.length, self.length)))
            self.apply_func(self.src, exp, self.predictions[-1])


    def show_prediction(self, n):
        m = np.max(self.predictions[n])
        img = Image.fromarray((255. / m * self.predictions[n]).astype(np.uint8))
        img.show()

    def show_hists(self, num_bins = 64):
        max_prob = max(np.max(prediction) for prediction in self.predictions)
        bins = map(lambda x: x * max_prob / num_bins, range(num_bins + 1))
        fig = plt.figure()
        max_level = int(log(self.length // 2, 2))
        num_experts = len(self.experts)
        results = np.zeros((max_level, num_experts))
        subplots = []
        count = 0
        for level in range(max_level-1, -1, -1):
            mask = level_mask(self.length, level)
            for pred in self.predictions:
                sp = fig.add_subplot(max_level, num_experts, count + 1)
                sp.hist(pred[mask], bins)
                subplots.append(sp)
                results[count // num_experts, count % num_experts] = -np.sum(defaultlog(pred[mask])) / mask.sum()
                count += 1
            ymax = max(sp.get_ylim() for sp in subplots[-num_experts:])
            for sp in subplots[-num_experts:]:
                sp.set_ylim(ymax)
            for sp in subplots[1 - num_experts:]:
                sp.set_yticklabels([])
        for sp in subplots[:-num_experts]:
            sp.set_xticklabels([])
        fig.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.05, wspace = 0.05, hspace = 0.05)
        fig.show()
        return fig, results

    def approx_cost(self, key):
        return -np.sum(np.log2(self.predictions[key][self.predictions[key] > 0]))

    def codelengths(self, key):
        cl = np.zeros((self.length, self.length))
        self.apply_func(self.src, partial(self.experts[key], cl=True), cl)
        return cl


    def mix(self):
        best = np.zeros((self.length, self.length, len(self.predictions)))
        for prediction, n in zip(self.predictions, range(len(self.predictions))):
            self.apply_func(prediction, lambda nbrs, pixel: sum(nbrs), best[:,:,n])
        mixed_prediction = np.zeros((self.length, self.length))
        for i in range(self.length):
            print i
            for j in range(self.length):
                mixed_prediction[i, j] = self.predictions[np.argmax(best[i, j])][i, j]
        return mixed_predictions



def level_mask(length, level):
    mask = np.zeros((length, length), dtype = bool)
    for i in range(length):
        for j in range(length):
            if (order2(i) == level and order2(j) == level) or (order2(i - j) == level and order2(i + j) == level):
                mask[i, j] = True
    return mask


def show_mask(length, level):
    mask = level_mask(length, level)
    img = Image.fromarray(255 * mask.astype(np.uint8))
    img.show()


def apply_func(src, func, dest):
    length = src.shape[0]
    level = length // 2
    while level > 0:
        print level
        for i in range(level, length, 2 * level):
            for j in range(level, length, 2 * level):
                nbr_indices = [(i - level, i - level, i + level, i + level),
                        (j - level, j + level, j - level, j + level)]
                pixel = src[i, j]
                nbrs = src[nbr_indices]
                dest[i, j] = func(nbrs, pixel)
        for i in range(0, length, level):
            if i % (2 * level) == 0:
                for j in range(level, length, 2 * level):
                    if i == 0:
                        nbr_indices = [(i + level, i, i), (j, j - level, j + level)]
                    elif i == length - 1:
                        nbr_indices = [(i - level, i, i), (j, j - level, j + level)]
                    else:
                        nbr_indices = [(i, i, i - level, i + level), (j - level, j + level, j, j)]
                    pixel = src[i, j]
                    nbrs = src[nbr_indices]
                    dest[i, j] = func(nbrs, pixel)
            else:
                for j in range(0, length, 2 * level):
                    if j == 0:
                        nbr_indices = [(i - level, i + level, i), (j, j, j + level)]
                    elif j == length - 1:
                        nbr_indices = [(i - level, i + level, i), (j, j, j - level)]
                    else:
                        nbr_indices = [(i, i, i - level, i + level), (j - level, j + level, j, j)]
                    pixel = src[i, j]
                    nbrs = src[nbr_indices]
                    dest[i, j] = func(nbrs, pixel)
        level /= 2



def level_mask(length, level):
    mask = np.zeros((length, length), dtype = bool)
    for i in range(length):
        for j in range(length):
            if (order2(i) == level and order2(j) == level) or (order2(i - j) == level and order2(i + j) == level):
                mask[i, j] = True
    return mask


def show_mask(length, level):
    mask = level_mask(length, level)
    img = Image.fromarray(255 * mask.astype(np.uint8))
    img.show()

        

def mix(predictions):
    length = predictions[0].shape[0]
    best = np.zeros((length, length, len(predictions)))
    for prediction, n in zip(predictions, range(len(predictions))):
        self.apply_func(prediction, lambda nbrs, pixel: sum(nbrs), best[:,:,n])
    mixed_prediction = np.zeros((length, length))
    for i in range(length):
        print i
        for j in range(length):
            mixed_prediction[i, j] = predictions[np.argmax(best[i, j])][i, j]
    return mixed_predictions

VG = [variable_gaussian]
L = [partial(mean_laplacian, x) for x in range(15, 26, 5)]
ML = [partial(mixture_of_laplacians, x) for x in range(5, 26, 5)]
U = [uniform]

path = "../test-images/test-images-513/018-0.png"
img = Image.open(path)
#H = Hierarchy(img, L + ML)
        

def mix(predictions):
    length = predictions[0].shape[0]
    performance = np.zeros((length, length, len(predictions)))
    for prediction, n in zip(predictions, range(len(predictions))):
        apply_func(prediction, lambda nbrs, pixel: sum(nbrs), performance[:,:,n])
    mixed_prediction = np.zeros((length, length))
    best = np.apply_along_axis(np.argmax, 2, performance)
    for i in range(length):
        print i
        for j in range(length):
            mixed_prediction[i, j] = predictions[best[i, j]][i, j]
    return mixed_prediction, best

VG = [variable_gaussian]
L = [partial(mean_laplacian, x) for x in range(15, 26, 5)]
ML = [partial(mixture_of_laplacians, x) for x in range(5, 26, 5)]
U = [uniform]

path = "../test-images/test-images-513/018-0.png"
img = Image.open(path)
#H = Hierarchy(img, L + ML)
