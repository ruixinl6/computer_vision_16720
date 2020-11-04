import numpy as np
def alignChannels(red, green, blue):
    
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""
    col,row = red.shape

    rgb = np.zeros([red.shape[0],red.shape[1],3])
    rgb[:,:,0] = red
    rgb[:,:,1] = green
    rgb[:,:,2] = blue

    ref = 200
    
    SSD_green_h = np.zeros([60,1])
    min_SSD_green_h = 256
    min_SSD_green_h_index = 0
    
    SSD_blue_h = np.zeros([60,1])
    min_SSD_blue_h = 256
    min_SSD_blue_h_index = 0
    
    for i in range(60):
        SSD_green_h[i] = (red[ref,:]-green[ref-30+i,:]).dot((red[ref,:]-green[ref-30+i,:]).T)
        if SSD_green_h[i] < min_SSD_green_h:
            min_SSD_green_h = SSD_green_h[i]
            min_SSD_green_h_index = i
        
        SSD_blue_h[i] = (red[ref,:]-blue[ref-30+i,:]).dot((red[ref,:]-blue[ref-30+i,:]).T)
        if SSD_blue_h[i] < min_SSD_blue_h:
            min_SSD_blue_h = SSD_blue_h[i]
            min_SSD_blue_h_index = i
        
    new_green = np.zeros([col,row])
    if min_SSD_green_h_index > 30:
        new_green[0:col-(min_SSD_green_h_index-30),:] = green[min_SSD_green_h_index-30:col,:]
    if min_SSD_green_h_index < 30:
        new_green[0:col+(min_SSD_green_h_index-30),:] = green[0:col+(min_SSD_green_h_index-30),:]
        
    new_blue = np.zeros([col,row])
    if min_SSD_blue_h_index > 30:
        new_blue[0:col-(min_SSD_green_h_index-30),:] = blue[min_SSD_green_h_index-30:col,:]
    if min_SSD_blue_h_index < 30:
        new_blue[0:col+(min_SSD_green_h_index-30),:] = blue[0:col+(min_SSD_green_h_index-30),:]
    
    rgb[:,:,1] = new_green
    rgb[:,:,2] = new_blue
    rgb = rgb.astype(np.uint8)

    return rgb