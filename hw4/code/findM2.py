'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper

data = np.load('../data/some_corresp.npz')
intrinsics = np.load('../data/intrinsics.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
pts1 = data['pts1']
pts2 = data['pts2']
K1 = intrinsics['K1']
K2 = intrinsics['K2']

N = pts1.shape[0]
M = 640

M1 = np.zeros((3,4))
M1[0,0] = 1
M1[1,1] = 1
M1[2,2] = 1
F = sub.eightpoint(pts1,pts2,M)
E = sub.essentialMatrix(F,K1,K2)
M2s = helper.camera2(E)
C1 = K1@M1

error = np.zeros((1,M2s.shape[2]))
for i in range(M2s.shape[2]):
    C2 = K2@M2s[:,:,i]
    P,err = sub.triangulate(C1,pts1,C2,pts2)
    error[:,i] = err
    if (np.where(P[:,2]>0)[0].shape[0]==N):
        P_final = P
        M2 = M2s[:,:,i]

C2 = K2@M2
P = P_final
np.savez('q3_3',M2,C2,P)        
    