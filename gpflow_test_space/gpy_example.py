import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPy

ap = np.linspace(1.2, 5.0, num = 10)
a = ap[:, np.newaxis]
b = 10*((1-np.exp(-a+2))**2)
c = np.column_stack((a,b))

df = pd.DataFrame(c, columns = ['r0', 'E'])
df.to_csv('sine.dat', sep=',', index=False, float_format='%12.12f')

k = GPy.kern.RBF(input_dim=1)
m = GPy.models.GPRegression(a, b, kernel=k)
m.optimize()
print(m.log_likelihood())
print(m)
