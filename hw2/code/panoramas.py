import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    outputL = 1600
    outputW = 800
    im2_w = cv2.warpPerspective(im2,H2to1,(outputL,outputW))
    pano_im = np.zeros([outputW,outputL,3])
    
    pano_im[0:(0+im1.shape[0]),0:im1.shape[1],:] = im1
    pano_im = np.maximum(pano_im,im2_w)
    pano_im = pano_im.astype(np.uint8)
            
            
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    width = 1200
    row = im2.shape[0]
    col = im2.shape[1]
    corner_2 = np.atleast_2d([[1,col,1,col],[1,1,row,row],[1,1,1,1]])
    corner_2w = H2to1@corner_2
    corner_2w = np.ceil(corner_2w/(corner_2w[2,:]))
    
    row_max = np.maximum(im1.shape[0],max(corner_2w[1,:]))
    row_min = np.minimum(0,min(corner_2w[1,:]))
    col_max = np.maximum(im1.shape[1],max(corner_2w[0,:]))
    col_min = np.minimum(0,min(corner_2w[0,:]))
    
    scale = (col_max-col_min)/(row_max-row_min)
    height = np.ceil(width/scale)
    outsize = (int(width),int(height))
    
    ratio = width/(col_max-col_min)
    M_scale = np.atleast_2d([[ratio,0,0],[0,ratio,0],[0,0,1]])
    M_tran = np.atleast_2d([[1,0,0],[0,1,-row_min],[0,0,1]])
    M = M_scale@M_tran
    im1_w = cv2.warpPerspective(im1,M,outsize)
    im2_w = cv2.warpPerspective(im2,np.matmul(M,H2to1),outsize)
    pano_im = np.maximum(im1_w,im2_w)
    pano_im = pano_im.astype(np.uint8)

    
    return pano_im

def generatePanorama(im1, im2):
    
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)

    return pano_im


if __name__ == '__main__':
# =============================================================================
#     im1 = cv2.imread('../data/incline_L.png')
#     im2 = cv2.imread('../data/incline_R.png')
#     print(im1.shape)
#     locs1, desc1 = briefLite(im1)
#     locs2, desc2 = briefLite(im2)
#     matches = briefMatch(desc1, desc2)
#     # plotMatches(im1,im2,matches,locs1,locs2)
#     H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
# =============================================================================
    pano_im = imageStitching(im1, im2, H2to1)
# =============================================================================
#     pano_im = imageStitching_noClip(im1, im2, H2to1)
#     im3 = generatePanorama(im1, im2)
# =============================================================================
    print(H2to1)
    cv2.imwrite('../results/6_1.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
