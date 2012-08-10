import hierarchy2
import cython_random_order as cro
from PIL import Image
import numpy as np
import pickle


predictions = []

codelengths = np.zeros((11, 6))


for i in range(1, 11):
    path = '../test-images/test-images-513/%03d-0.png' % i
    img = Image.open(path)
    arr = np.asarray(img, dtype = np.uint8)
    pred = [cro.process(arr, num_nbrs = 4), 
            cro.process(arr, num_nbrs = 8), 
            cro.process(arr, num_nbrs = 16), 
            cro.process(arr, num_nbrs = 4, random_order = True), 
            cro.process(arr, num_nbrs = 8, random_order = True), 
            cro.process(arr, num_nbrs = 16, random_order = True)]
    predictions.append(pred)
    for j in range(6):
        cl = -np.sum(np.log2(pred[j][pred[j] > 0]))
        print cl
        codelengths[i - 1, j] = cl

f = open("codelength_results", 'w')
pickle.dump(codelengths, f)
f.close()
