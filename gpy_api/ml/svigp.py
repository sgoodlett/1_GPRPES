from .gaussian_process import GaussianProcess
from GPy.core.svgp import SVGP
from GPy.kern import RBF
from GPy.likelihoods.gaussian import Gaussian
from climin import Adadelta
import numpy as np

class SVIGP(GaussianProcess):

    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, train_path=None, test_path=None, batchsize = 10, inducing_n = 10, max_iter = 5000):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path)
        self.batchsize = batchsize
        self.inducing_n = inducing_n
        self.max_iter = max_iter
        if self.inducing_n > self.train_indices:
            print("Woah there Nelly, yall can't be askin' for more inducing inputs than training inputs. Better back on up while we set them equal to each other.")
            self.inducing_n = self.train_indices
        if self.batchsize >= self.train_indices:
            raise Exception("If you are going to set the batchsize to be greater than or equal to the number of training inputs, might I suggest you do some more reading before using SVIGP?")
        self.choose_inducing_inputs()

    def build_model(self, params)
        print("Hyperparameters:",params)
        self.split_train_test(params)
        dim = self.X.shape[1]
        if self.input_obj.keywords['gp_ard'] == 'opt':
            ard_val = params['ARD']
        elif self.input_obj.keywords['gp_ard'] == 'true':
            ard_val = True
        else:
            ard_val = False
        kernel = RBF(dim, ARD=ard_val)  # TODO add HP control of kernel
        self.model = GPy.core.svgp.SVGP(self.Xtr, self.ytr, self.Z, kernel, Gaussian(), batchsize=self.batchsize)
        opt = Adadelta(self.model.optimizer_array, self.model.stochastic_grad, step_rate=0.2, momentum=0.9)
        #self.model = GPRegression(self.Xtr, self.ytr, kernel=kernel, normalizer=False)
        #self.model.optimize_restarts(nrestarts, optimizer="lbfgsb", robust=True, verbose=False, max_iters=maxit, messages=False)
        gc.collect(2) #fixes some memory leak issues with certain BLAS configs

    def callback(self, i):
        if i['n_iter'] > self.max_iter:
            return True
        return False

    def choose_inducing_inputs(self):
        idx_subset = np.random.choice(self.train_indices, size = self.inducing_n, replace=False)
        self.Z = self.Xtr
