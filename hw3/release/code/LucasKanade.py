import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage


def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    # Put your implementation here
    p = p0

# =============================================================================
#     x1 = rect[0,0]
#     y1 = rect[1,0]
#     x2 = rect[2,0]
#     y2 = rect[3,0]
# =============================================================================
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]


    x = np.linspace(x1,x2,(x2-x1+1))
    y = np.linspace(y1,y2,(y2-y1+1))
    X,Y = np.meshgrid(x,y)
    row = int(y2-y1+1)
    col = int(x2-x1+1)
    h = It.shape[0]
    w = It.shape[1]
    
    
    It_spline = RectBivariateSpline(np.linspace(0,h-1,h), np.linspace(0,w-1,w), It)
    It1_spline = RectBivariateSpline(np.linspace(0,h-1,h), np.linspace(0,w-1,w), It1)

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
        H = ((jacobian*sd).T)@(jacobian*sd)
        
        delta_p = ((jacobian*sd).T)@error.flatten()
        delta_p = np.linalg.inv(H).dot(delta_p)
        
        print('current delta is: ',delta_p)
        p = p+delta_p

    return p
