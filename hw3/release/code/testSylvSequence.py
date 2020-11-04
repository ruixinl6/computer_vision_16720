import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeBasis import LucasKanadeBasis
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
data = np.load('../data/sylvseq.npy')
bases = np.load('../data/sylvbases.npy')
# =============================================================================
# cv2.imshow('frame',data[:,:,20])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================
num_frame = data.shape[2]
rects = np.zeros([num_frame,4])
rects[0,:] = np.atleast_2d([101,61,155,107])
rects2 = rects

for i in range(0,num_frame-1):
    It = data[:,:,i]
    It1 = data[:,:,i+1]
    rect = np.atleast_2d(rects[i,:]).T
    p = LucasKanadeBasis(It,It1,rect,bases)
    rects[i+1,:] = np.atleast_2d([rects[i,0]+p[0],rects[i,1]+p[1],rects[i,2]+p[0],rects[i,3]+p[1]])
    
for i in range(0,num_frame-1):
    It = data[:,:,i]
    It1 = data[:,:,i+1]
    rect = np.atleast_2d(rects[i,:]).T
    p = LucasKanade(It,It1,rect,p0 = np.zeros(2))
    rects2[i+1,:] = np.atleast_2d([rects2[i,0]+p[0],rects2[i,1]+p[1],rects2[i,2]+p[0],rects2[i,3]+p[1]])
    
    
    
for i in [0,199,299,349,399]:
    # Create figure and axes
    fig,ax = plt.subplots()
    # Display the image
    ax.imshow(data[:,:,i])
    # Create a Rectangle patch
    
    rect2 = patches.Rectangle((rects2[i,0],rects2[i,1]),(rects2[i,2]-rects2[i,0]),(rects2[i,3]-rects2[i,1]),linewidth=2,edgecolor='r',facecolor='none')
    rect1 = patches.Rectangle((rects[i,0],rects[i,1]),(rects[i,2]-rects[i,0]),(rects[i,3]-rects[i,1]),linewidth=2,edgecolor='b',facecolor='none')
    # Add the patch to the Axes
    
    ax.add_patch(rect2)
    ax.add_patch(rect1)
    plt.show()
    fig.savefig('image_Q23_'+str(i)+'.png')
    
np.save('sylvseqrects.npy',rects)