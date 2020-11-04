import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    im = skimage.color.rgb2gray(image)
    im = skimage.filters.gaussian(im)
    threshold = skimage.filters.otsu(im)
    Bin = im<threshold
    opening = skimage.morphology.binary_opening(Bin)
    labels = skimage.measure.label(opening)
    props = skimage.measure.regionprops(labels)
    for prop in props:
        if prop.area > 500 and prop.area < 15000:
            bboxes.append(prop.bbox)
    bw = 1-opening
    
    
    
    
    
    
    
    
    
    
    return bboxes, bw