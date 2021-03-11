import numpy as np
import pandas as pd
import gpflow as gpf
import matplotlib.pyplot as plt
import tensorflow as tf

ap = np.linspace(1.2, 5.0, num = 10)
a = ap[:, np.newaxis]
b = 10*((1-np.exp(-a+2))**2)
c = np.column_stack((a,b))

df = pd.DataFrame(c, columns = ['r0', 'E'])
df.to_csv('sine.dat', sep=',', index=False, float_format='%12.12f')

k = gpf.kernels.RBF()
gpf.utilities.print_summary(k)
m = gpf.models.GPR(data=(a,b), kernel=k)
m.kernel.lengthscales.assign(0.3)
m.likelihood.variance.assign(0.1)
gpf.utilities.print_summary(m)
opt = gpf.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables)
gpf.utilities.print_summary(m)

xx = np.linspace(0.0, 7.0, num = 100).reshape(100,1)
mean, var = m.predict_f(xx)

tf.random.set_seed(1)  # for reproducibility
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

plt.figure(figsize=(12, 6))
plt.plot(a, b, "kx", mew=2)
plt.plot(xx, mean, "C0", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)

plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
_ = plt.xlim(0.0, 7.0)
plt.show()
print(m.log_marginal_likelihood())
