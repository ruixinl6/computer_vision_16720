import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    length = p1.shape[1]
    u1 = np.atleast_2d(p1[0,:]).T
    v1 = np.atleast_2d(p1[1,:]).T
    x2 = np.atleast_2d(p2[0,:]).T
    y2 = np.atleast_2d(p2[1,:]).T
    
    A = np.zeros([2*length,9])
    index = np.arange(length)
    
    A1 = np.hstack([x2,y2,np.ones([length,1]),np.zeros([length,3]),-x2*u1,-y2*u1,-u1])
    A2 = np.hstack([np.zeros([length,3]),x2,y2,np.ones([length,1]),-x2*v1,-y2*v1,-v1])
    A[2*index,:] = A1
    A[2*index+1,:] = A2
    
    e_value,e_vector = np.linalg.eigh((A.T).dot(A))
    index2 = np.argmin(np.abs(e_value))
    h = e_vector[:,index2]
    H2to1 = h.reshape(3,3)
    
    
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    p1 = locs1[matches[:,0],0:2].T
    p2 = locs2[matches[:,1],0:2].T
    length = matches.shape[0]
    inlier_max = 0

    
    p1s = np.vstack((p1,np.ones([1,length])))
    p2s = np.vstack((p2,np.ones([1,length])))
    
    for i in range(0,num_iter):
        index = np.random.choice(length, 4,replace=False)
        p1_train = p1[:,index]
        p2_train = p2[:,index]
        H = computeH(p1_train,p2_train)
        
        p1s_H = H.dot(p2s)
        p1s_H = p1s_H/p1s_H[2,:]
        diff = p1s-p1s_H
        error = np.linalg.norm(diff,axis=0)
        inlier = (error<tol)
        if np.sum(inlier) > np.sum(inlier_max):
            inlier_max = inlier
            p1_final = np.squeeze(p1[:,np.asarray(np.where(inlier==1))])
            p2_final = np.squeeze(p2[:,np.asarray(np.where(inlier==1))])
            
    bestH = computeH(p1_final,p2_final)
    return bestH

        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

