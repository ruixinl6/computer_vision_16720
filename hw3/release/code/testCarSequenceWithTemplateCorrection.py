import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
data = np.load('../data/carseq.npy')
# =============================================================================
# cv2.imshow('frame',data[:,:,20])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================
num_frame = data.shape[2]
rects = np.zeros([num_frame,4])
rects[0,:] = np.atleast_2d([59,116,145,151])
It0 = data[:,:,0]

for i in range(0,num_frame-1):
    It = data[:,:,i]
    It1 = data[:,:,i+1]
    rect = np.atleast_2d(rects[i,:]).T
    p = LucasKanade(It,It1,rect,p0 = np.zeros(2))
    rects[i+1,:] = np.atleast_2d([rects[i,0]+p[0],rects[i,1]+p[1],rects[i,2]+p[0],rects[i,3]+p[1]])
    
    p_0_n = rects[i,0:2]-rects[0,0:2]
    
    p_0_nplus1 = LucasKanade(It0,It1,rects[0,:],p0 = p_0_n)
    delta = p_0_nplus1-p_0_n-p
    
    if np.linalg.norm(delta)<3:
        rects[i+1,:] = np.atleast_2d([rects[i+1,0]+delta[0],rects[i+1,1]+delta[1],rects[i+1,2]+delta[0],rects[i+1,3]+delta[1]])


rects2 = np.load('carseqrects.npy')

for i in range(0,5):
    # Create figure and axes
    fig,ax = plt.subplots()
    # Display the image
    ax.imshow(data[:,:,i*100])
    # Create a Rectangle patch
    rectangle = patches.Rectangle((rects[i*100,0],rects[i*100,1]),(rects[i*100,2]-rects[i*100,0]),(rects[i*100,3]-rects[i*100,1]),linewidth=1,edgecolor='r',facecolor='none')
    rectangle2 = patches.Rectangle((rects2[i*100,0],rects2[i*100,1]),(rects2[i*100,2]-rects2[i*100,0]),(rects2[i*100,3]-rects2[i*100,1]),linewidth=1,edgecolor='b',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rectangle)
    ax.add_patch(rectangle2)
    plt.show()
    fig.savefig('image_Q14_'+str(i)+'.png')
    
np.save('carseqrects-wcrt.npy',rects)