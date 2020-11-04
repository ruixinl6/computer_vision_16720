import numpy as np
import cv2
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from matplotlib import pyplot as plt

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_pyramid = np.zeros([gaussian_pyramid.shape[0],gaussian_pyramid.shape[1],gaussian_pyramid.shape[2]-1])
    DoG_levels = levels[1:]
    for i in DoG_levels:
        DoG_pyramid[:,:,i] = gaussian_pyramid[:,:,i+1]-gaussian_pyramid[:,:,i]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    ##################
    # TO DO ...
    # Compute principal curvature here

    Dx = cv2.Sobel(DoG_pyramid,cv2.CV_64F,1,0,ksize=1)
    Dy = cv2.Sobel(DoG_pyramid,cv2.CV_64F,0,1,ksize=1)
    Dxx = cv2.Sobel(Dx,cv2.CV_64F,1,0,ksize=1)
    Dxy = cv2.Sobel(Dx,cv2.CV_64F,0,1,ksize=1)
    Dyx = cv2.Sobel(Dy,cv2.CV_64F,1,0,ksize=1)
    Dyy = cv2.Sobel(Dy,cv2.CV_64F,0,1,ksize=1)
    Tr = Dxx+Dyy
    det = Dxx*Dyy-Dxy*Dyx
    principal_curvature = Tr**2/det
    
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None
    ##############
    #  TO DO ...
    # Compute locsDoG here
    window = np.zeros([3,3,3])
    window[:,:,1] = 1
    window[1,1,:] = 1
    data_max = filters.maximum_filter(DoG_pyramid, footprint=window)
    maxima = (DoG_pyramid == data_max)
    data_min = filters.minimum_filter(DoG_pyramid, footprint=window)
    minima = (DoG_pyramid == data_min)
    coords = np.where((maxima|minima)&(np.abs(DoG_pyramid)>th_contrast)&(principal_curvature<th_r))
    locsDoG = np.vstack((coords[1],coords[0],coords[2])).T
    
    
    
    
    return locsDoG

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    im_pyr = createGaussianPyramid(im)
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    
    return locsDoG, im_pyr







if __name__ == '__main__':
    # test gaussian pyramid
    #levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    #im_pyr = createGaussianPyramid(im)
    #displayPyramid(im_pyr)
    # test DoG pyramid
    #DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    #displayPyramid(DoG_pyr)
    # test compute principal curvature
    #pc_curvature = computePrincipalCurvature(DoG_pyr)
    # displayPyramid(pc_curvature)
    # test get local extrema
    #th_contrast = 0.03
    #th_r = 12
    #locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
    #cv2.imshow('image',im[:,:,0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #fig = plt.figure(1)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    for i in range(np.size(locsDoG,axis=0)):
        cv2.circle(im,(locsDoG[i,0],locsDoG[i,1]),1,(0,0,255),-1)
    cv2.imwrite('../results/q1_5.jpg', im)
    cv2.imshow('pig',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #plt.plot(locsDoG[:,1],locsDoG[:,0],'bo',markersize=2)


