import numpy as np
import pandas as pd
import GPy
from climin import Adadelta
import matplotlib.pyplot as plt

ap = np.linspace(1.2, 5.0, num = 1000000)
a = ap[:, np.newaxis]
b = 10*((1-np.exp(-a+2))**2)
c = np.column_stack((a,b))

df = pd.DataFrame(c, columns = ['r0', 'E'])
df.to_csv('sine.dat', sep=',', index=False, float_format='%12.12f')

Z = np.random.choice(ap, size = 8, replace = False)
Z = Z[:, np.newaxis]
m = GPy.core.SVGP(a, b, Z, GPy.kern.RBF(1) + GPy.kern.White(1), GPy.likelihoods.Gaussian(), batchsize = 10)
m.kern.white.variance = 1e-5
m.kern.white.fix()

#opt = Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.2, momentum = 0.9)
def callback(i):
    print(m.log_likelihood())
    if i['n_iter'] > 5000:
        return True
    return False
#info = opt.minimize_until(callback)

m.optimizeWithFreezingZ()

fig, ax = plt.subplots(1,1)
ax.plot(a,b, 'ko')
_ = m.plot()
plt.show()
