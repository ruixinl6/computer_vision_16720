import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    p0 = np.zeros(2)
    x1 = int(rect[0,0])
    y1 = int(rect[1,0])
    x2 = int(rect[2,0])
    y2 = int(rect[3,0])
    y = np.arange(y1,y2+1)
    x = np.arange(x1,x2+1)
    X,Y = np.meshgrid(x,y)
    row = It[y1:(y2+1),x1:(x2+1)].shape[0]
    col = It[y1:(y2+1),x1:(x2+1)].shape[1]
    
    
    It_spline = RectBivariateSpline(y, x, It[y1:(y2+1),x1:(x2+1)])
    It1_spline = RectBivariateSpline(y, x, It1[y1:(y2+1),x1:(x2+1)])
    
    num_base = bases.shape[2]

    p = p0
    delta_p = np.atleast_2d([1,1]).T
    threshold = 0.01
    
    while np.linalg.norm(delta_p)>threshold:
        It1_shifted = It1_spline.ev(Y+p[1],X+p[0]).reshape((row,col))
        error = It_spline.ev(Y,X).reshape((row,col)) - It1_shifted  
        
        It1_shifted_x = It1_spline.ev(Y+p[1],X+p[0],dx=0,dy=1).reshape((row,col))
        It1_shifted_y = It1_spline.ev(Y+p[1],X+p[0],dx=1,dy=0).reshape((row,col))
        sd = np.vstack((It1_shifted_x.flatten(),It1_shifted_y.flatten())).T
        jacobian = 1
        
        for i in range(num_base):
            base = bases[:,:,i]
            inner_part = base.flatten()@sd
            inner_part = base.reshape((base.shape[0]*base.shape[1],1))@np.atleast_2d(inner_part)
            sd = sd - inner_part
            
# =============================================================================
#         base = np.reshape(bases,((bases.shape[0]*bases.shape[1],bases.shape[2])))
#         inner_part = base.T@sd
#         inner_part = inner_part.T@base.T
#         sd = sd - inner_part
# =============================================================================
        
        H = ((jacobian*sd).T)@(jacobian*sd)
        
        delta_p = ((jacobian*sd).T)@error.flatten()
        delta_p = np.linalg.inv(H).dot(delta_p)
        
        print('current delta is: ',delta_p)
        p = p+delta_p
    
    
    
    
    
    return p
    
