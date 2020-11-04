"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper as helper
import scipy.ndimage as ndimage
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = pts1/M
    pts2 = pts2/M
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    length = x2.shape[0]
    
    u = np.vstack((x2*x1,x2*y1,x2,y2*x1,y2*y1,y2,x1,y1,np.ones(length))).T
    _,_,V = np.linalg.svd(u.T@u)
    F = V[-1,:].reshape((3,3))
    
    U2,S2,V2 = np.linalg.svd(F)
    S2_new = np.zeros((3,3))
    S2_new[0,0] = S2[0]
    S2_new[1,1] = S2[1]
    F = U2@S2_new
    F = F@(V2)
    F = helper.refineF(F,pts1,pts2)
    
    T = np.atleast_2d([[1/M,0,0],[0,1/M,0],[0,0,1]])
    F_unnorm = T.T@F
    F_unnorm = F_unnorm@T
    
    
    
    return F_unnorm


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = pts1/M
    pts2 = pts2/M
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    length = x2.shape[0]
    
    u = np.vstack((x2*x1,x2*y1,x2,y2*x1,y2*y1,y2,x1,y1,np.ones(length))).T
    _,_,V = np.linalg.svd(u.T@u)
    F1 = V[-1,:].reshape((3,3)).T
    F2 = V[-2,:].reshape((3,3)).T
    
    fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
    a0 = fun(0)
    a1 = 2*(fun(1)-fun(-1))/3-(fun(2)-fun(-2))/12
    a2 = 0.5*fun(1)+0.5*fun(-1)-fun(0)
    a3 = (fun(2)-fun(-2))/12-(fun(1)-fun(-1))/6
    root = np.roots([a3,a2,a1,a0])
    is_complex = np.iscomplex(root)
    Lambda = np.real(root[np.where(is_complex==0)])
    
    F = [None]*Lambda.shape[0]
    
    T = np.atleast_2d([[1/M,0,0],[0,1/M,0],[0,0,1]])
    for i in range(Lambda.shape[0]):
        F[i] = Lambda[i]*F1+(1-Lambda[i])*F2
        F[i] = helper.refineF(F[i],pts1,pts2)
        F[i] = T.T@F[i]
        F[i] = F[i]@T
        
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K2.T@F
    E = E@K1
    
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    P = np.zeros((x2.shape[0],3))
    
    for i in range(x2.shape[0]):
        A = [[x1[i]*C1[2,:]-C1[0,:]],[y1[i]*C1[2,:]-C1[1,:]],
             [x2[i]*C2[2,:]-C2[0,:]],[y2[i]*C2[2,:]-C2[1,:]]]
        A = np.atleast_2d(A).squeeze()
        _,_,V = np.linalg.svd(A)
        P_h = V[-1,:]
        P[i,:] = P_h[0:3]/P_h[3]
    
    P_H = np.vstack((P.T,np.ones((1,x2.shape[0]))))
    p_1 = C1@P_H
    p_1 = p_1[0:2,:]/np.vstack((p_1[2,:],p_1[2,:]))
    p_2 = C2@P_H
    p_2 = p_2[0:2,:]/np.vstack((p_2[2,:],p_2[2,:]))

    err = np.sum((p_1.T-pts1)**2+(p_2.T-pts2)**2)
    
    return P,err
    


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    p1 = np.array([x1,y1,1]).T
    epipolarLine = F@p1
    epipolarLine = epipolarLine/(np.linalg.norm(epipolarLine))
    a1 = epipolarLine[0]
    a2 = epipolarLine[1]
    a3 = epipolarLine[2]
    
    x1 = int(x1)
    y1 = int(y1)
    sigma = 4
    width = 10
    window1 = im1[(y1-width):(y1+width),(x1-width):(x1+width)]
    
    error_min = 100000000
    for y2_candidate in np.arange(y1-sigma*width,y1+sigma*width):
        x2_candidate = (-a2*y2_candidate-a3)/a1
        x2_candidate = int(x2_candidate)
        
        if(x2_candidate-width)>0 and (x2_candidate+width<im2.shape[1]) and (y2_candidate-width)>0 and (y2_candidate+width)<im2.shape[0]:
            window2 = im2[(y2_candidate-width):(y2_candidate+width),(x2_candidate-width):(x2_candidate+width)]
            dist = window1-window2
            dist = ndimage.gaussian_filter(dist,sigma)
            error = (np.sum(dist**2))**(1/2)
            
            if error < error_min:
                error_min = error
                x2 = x2_candidate
                y2 = y2_candidate
                
    return x2,y2
        
    
    
    
