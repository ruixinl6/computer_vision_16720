from alignChannels import alignChannels
from matplotlib import pyplot as plt
import scipy.misc
# Problem 1: Image Alignment
import numpy as np
# 1. Load images (all 3 channels)
red = np.load('data/red.npy')
green = np.load('data/green.npy')
blue = np.load('data/blue.npy')
# 2. Find best alignment
#print(red.shape[0])
rgbResult = alignChannels(red, green, blue)
plt.imshow(rgbResult)
plt.show()


# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)
scipy.misc.imsave('rgb_output.jpg', rgbResult)









