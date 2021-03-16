import numpy as np
import tensorflow as tf
import gpflow
from .gaussian_process import GaussianProcess

class SVIGP(GaussianProcess):
    

    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, train_path=None, test_path=None):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path)

    def build_model(self, params, nrestarts=10, maxiter=1000, seed=0):
        print("Hyperparameters: ", params)
        self.split_train_test(params)
        np.random.seed(seed)     # make GPy deterministic for a given hyperparameter config
        #TODO: ARD
        self.model = self.opt_with_restarts(self.opt_res_fxn, iters=nrestarts, maxiter=maxiter)
        gpflow.utilities.print_summary(self.model)
        gc.collect(2) #fixes some memory leak issues with certain BLAS configs

    def opt_with_restarts(self, fxn, iters, maxiter):
        # Optimize model several times with random parameter initiation. This will hopefully bypass the issue with Cholesky Decomposition
        models = []
        for i in range(iters):
            model_i, opt_i = fxn()
            try:
                # Do an opt
                logs = opt_i.minimize(model_i.training_loss, model_i.trainable_variables, options = dict(maxiter=maxiter))
            
            except tf.errors.InvalidArgumentError:
                print("Optimization went wild, moving on to the next iteration. This is why we do restarts.")
                
            else:
                # Wrap things up
                models.append(model_i)
        return sorted(models, key = lambda x: x.log_marginal_likelihood())[-1]

    def opt_res_fxn(self):
        # Defines how to initialize model params and restart opt
        r = np.random.rand(3)
        kernel = gpflow.kernels.RBF(variance = (r[0]*100), lengthscales = (r[1]*10)) + gpflow.kernels.White(variance = (r[2]*0.0001))
        model = gpflow.models.GPR(data = (self.Xtr, self.ytr), kernel = kernel)
        opt = gpflow.optimizers.Scipy()
        return model, opt

