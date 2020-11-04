import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    p0 = np.zeros(6)
    row = It.shape[0]
    col = It.shape[1]
    x = np.arange(0,col)
    y = np.arange(0,row)
    X,Y = np.meshgrid(x,y)
    
    It_spline = RectBivariateSpline(y, x, It)
    
    It_shifted_x = It_spline.ev(Y,X,dx=0,dy=1)
    It_shifted_y = It_spline.ev(Y,X,dx=1,dy=0)
    sd = np.vstack((It_shifted_x*X,
                    It_shifted_y*X,
                    It_shifted_x*Y,
                    It_shifted_y*Y,
                    X,Y)).reshape((row,col,6))
    H = sd.reshape((row*col,6)).T@sd.reshape((row*col,6)) 
    
    delta_p = np.atleast_2d([1,1,1,1,1,1]).T
    threshold = 1
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = p0
     
    while np.linalg.norm(delta_p)>threshold:
        common_area = np.ones((row,col))
        common_area = ndimage.affine_transform(common_area,M)
        y_used,x_used = np.where(common_area!=0)
        
        It1_warp = ndimage.affine_transform(It1,M)
        It1_warp = It1_warp[y_used,x_used]
        It_new = It[y_used,x_used]
        err = It1_warp - It_new
        
        sd_used = sd[y_used,x_used,:]
        b = sd_used.T@err
        delta_p = np.linalg.inv(H)@b
        p = p-delta_p
        M = np.array([[1.0+p[0], 0.0+p[2], 0.0+p[4]], [0.0+p[1], 1.0+p[3], 0.0+p[5]]])
        
    return M