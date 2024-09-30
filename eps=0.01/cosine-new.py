#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import time
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel
from sklearn.gaussian_process.kernels import Hyperparameter
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform

np.random.seed(2024)


# In[2]:


class Cosine(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Cosine kernel.

    Parameters
    ----------
    period : float or ndarray of shape (n_features,), default=1.0
        The period of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    period_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'period'.
        If set to "fixed", 'period' cannot be changed during
        hyperparameter tuning.
    """

    def __init__(self, period=1.0, period_bounds=(1e-5, 1e5)):
        self.period = period
        self.period_bounds = period_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.period) and len(self.period) > 1

    @property
    def hyperparameter_period(self):
        if self.anisotropic:
            return Hyperparameter(
                "period",
                "numeric",
                self.period_bounds,
                len(self.period),
            )
        return Hyperparameter("period", "numeric", self.period_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        period = self.period
        if Y is None:
            # Compute pairwise differences
            diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            K = np.cos(2 * np.pi * diffs / period)
            # Sum over the feature dimensions
            K = np.sum(K, axis=-1)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            # Compute pairwise differences
            diffs = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
            K = np.cos(2 * np.pi * diffs / period)
            # Sum over the feature dimensions
            K = np.sum(K, axis=-1)

        if eval_gradient:
            if self.hyperparameter_period.fixed:
                # Hyperparameter period kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or period.shape[0] == 1:
                # Gradient computation for isotropic case
                K_gradient = 2 * np.pi * np.sin(2 * np.pi * diffs / period) * diffs / period
                K_gradient = np.sum(K_gradient, axis=-1)[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # Gradient computation for anisotropic case
                K_gradient = 2 * np.pi * np.sin(2 * np.pi * diffs / period) * (diffs / period[:, np.newaxis])
                K_gradient = np.sum(K_gradient, axis=-1)[:, :, np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(period=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.period)),
            )
        else:  # isotropic
            return "{0}(period={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.period)[0]
            )


# In[15]:


def construct_cosine_kernel(params):
    kernel = params[0] * Cosine(period=params[1], period_bounds = (params[1] / 2, 2 * params[1]))
    return kernel

def simulate_mles(n_max, ns, true_params, initial_params, eps, num_restarts=0):
    # Create the x array with shifts
    jitter = 1 / (4 * n_max)
    x = np.linspace(start=jitter, stop=1-jitter, num=n_max).reshape(-1, 1)
    shift = np.random.uniform(-jitter, jitter, size=n_max).reshape(-1, 1)
    x = x + shift
    
    # Define the true kernel and generate y
    true_kernel = construct_cosine_kernel(true_params) + WhiteKernel(noise_level=eps)
    true_gp = GaussianProcessRegressor(kernel=true_kernel, alpha=0)
    
    # y = np.squeeze(true_gp.sample_y(x, random_state=None))
    l = np.linalg.cholesky(true_kernel(x))
    y = np.dot(l, np.random.normal(size=len(x)))
    
    mles = []
    for n in ns:
        # Construct the initial kernel
        kernel = construct_cosine_kernel(initial_params)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=num_restarts, alpha=eps)
        
        # Select a random subset of x and y
        subset_indices = np.random.choice(n_max, size=n, replace=False)
        subset_x = x[subset_indices]
        subset_y = y[subset_indices]
        
        # Fit the GP model
        gp.fit(subset_x, subset_y)
        
        # Extract the fitted kernel parameters
        gp_params = gp.kernel_.get_params()
        param_estimates = [gp_params['k1__constant_value'],
                           gp_params['k2__period'],
                          ]
        
        # Append the estimated parameters to mles
        mles.append(param_estimates)
    
    # Return the estimated parameters
    return mles

# Parameters
n_max = 1000
ns = [1000]
true_params = [1, 1/4]
initial_params = [1 * p for p in true_params]
eps = 0.01

# Run the simulation and time it
start_time = time.time()
estimated_params = simulate_mles(n_max, ns, true_params, initial_params, eps, num_restarts=0)
end_time = time.time()
total_time = end_time - start_time

# Print the results
print("Estimated parameters for different sample sizes:")
for i, n in enumerate(ns):
    print("n = {}: {}".format(n, estimated_params[i]))

print("Total simulation time: {:.4f} seconds".format(total_time))


# In[11]:


kernel = construct_cosine_kernel([1, 1/4])
gp_params = kernel.get_params()
gp_params


# In[16]:


# Calculate mles and save to a file
def get_param_estimates(n_max, ns, true_params, initial_params, eps, num_restarts, num_replicates):
    # Collect estimates
    estimates = []
    for _ in range(num_replicates):
        estimates.append(simulate_mles(n_max, ns, true_params, initial_params, eps, num_restarts=0))
    
    param_estimates = []
    for i in range(len(true_params)):
        ith_param_estimates = []
        for j in range(len(ns)):
            ith_param_jth_sample_size_param_estimates = []
            for replicate in estimates:
                ith_param_jth_sample_size_param_estimates.append(replicate[j][i])
            ith_param_estimates.append(ith_param_jth_sample_size_param_estimates)
        param_estimates.append(ith_param_estimates)
    np.save('./results/cosine-simulation.npy', param_estimates)
    return param_estimates

# In[56]:


# Timing execution
start_time = time.time()

# Parameters
num_replicates = 100
true_params = [1, 1 / 4]  # Example true parameters
initial_params = [1 * p for p in true_params]  # Example initial parameters
eps = 0.01
num_restarts = 0
n_max = 5000
ns = [500, 1000, 2000, 5000]  # Example sample sizes

# Generate plots
np.random.seed(2024)
param_estimates = get_param_estimates(n_max, ns, true_params, initial_params, eps, num_restarts, num_replicates)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")


# In[ ]:




