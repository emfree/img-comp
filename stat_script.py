import patch_statistics as pt
import pickle
import matplotlib.pyplot as plt


f = open("top_10_thresholded_patches.txt")
T = pickle.load(f)
distances, s_data = pt.array_stats(T)


