import cython_random_order as cro
import numpy as np
from PIL import Image
img = np.asarray(Image.open("../test-images/test-images-513/001-0.png"), dtype = int)
arr = img[256:, 256:]
mvl_prob, mvl_var = cro.process(arr, showerr = True)
