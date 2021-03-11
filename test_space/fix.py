import numpy as np

X = np.asarray([[1,2],[2,3],[3,4],[4,5],[5,6]])
Y = np.asarray([[0],[1],[2],[3],[4]])

h = np.asarray(range(X.shape[0]))
print(h)

i = np.random.choice(h,size=3,replace=False)
print(i)

print(Y[i])
