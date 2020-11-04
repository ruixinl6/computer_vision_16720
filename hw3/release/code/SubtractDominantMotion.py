import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from scipy import ndimage
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    row = image1.shape[0]
    col = image1.shape[1]
    
    
    M = InverseCompositionAffine(image1, image2)
    It_warp = ndimage.affine_transform(image1,M,(col,row))
    i_unused_y, i_unused_x = np.where(It_warp==0)
    It1_new = image2
    It1_new[i_unused_y, i_unused_x] = 0
    
    
    sub = np.absolute(It1_new-It_warp)
    mask_x, mask_y = np.where(sub < 0.15)
    mask = np.ones(image1.shape, dtype=bool)
    mask[mask_x, mask_y] = 0
    
    return mask
