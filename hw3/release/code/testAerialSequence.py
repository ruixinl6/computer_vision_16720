import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
import time


# write your script here, we recommend the above libraries for making your animation
start = time.clock()
data = np.load('../data/aerialseq.npy')
num_frame = data.shape[2]
It0 = data[:,:,0]
mask = np.ones((It0.shape[0],It0.shape[1],num_frame))
for i in range(0,num_frame-1):
    It = data[:,:,i]
    It1 = data[:,:,i+1]
    mask[:,:,i] = SubtractDominantMotion(It,It1)
mask[0:5,:,:] = 0
mask[:,0:60,:] = 0
mask[180:,:,:] = 0
mask[:,250:,:] = 0 #eliminate regions where only noise exists.
end = time.clock()
print('Time elapsed: ',end-start)

for i in range(1,5):
    # Create figure and axes
    fig,ax = plt.subplots()
    # Display the image
    ax.imshow(data[:,:,i*30-1])
    mask_curr = mask[:,:,i*30-1]
    obj_y,obj_x = np.where(mask_curr==1)
    ax.scatter(obj_x,obj_y,color='r',s=1)
    
    plt.show()
    fig.savefig('image_Q33_'+str(i)+'.png')
