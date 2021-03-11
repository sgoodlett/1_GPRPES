import numpy as np

A = np.array([[0,1,2],[1,2,3],[2,3,4]])
C = np.array([[1,0,0],[0,1,0],[0,0,1]])
B = np.random.multivariate_normal([1,2,3],C,3)

print(B)
print(B[0,:])
D = np.hstack((A,B[0,:][:,None]))
print(D)
