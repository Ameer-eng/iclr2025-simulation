#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import scipy
import sklearn
import time
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

np.random.seed(2024)


# In[59]:


def construct_RBF_kernel(params):
    kernel = params[0] * RBF(length_scale=params[1])
    return kernel

def simulate_mles(n_max, ns, true_params, initial_params, eps, num_restarts=0):
    # Create the x array with shifts
    jitter = 1 / (4 * n_max)
    x = np.linspace(start=jitter, stop=1-jitter, num=n_max).reshape(-1, 1)
    shift = np.random.uniform(-jitter, jitter, size=n_max).reshape(-1, 1)
    x = x + shift
    
    # Define the true kernel and generate y
    true_kernel = construct_RBF_kernel(true_params) + WhiteKernel(noise_level=eps)
    true_gp = GaussianProcessRegressor(kernel=true_kernel, alpha=0)
    
    # y = np.squeeze(true_gp.sample_y(x, random_state=None))
    l = np.linalg.cholesky(true_kernel(x))
    y = np.dot(l, np.random.normal(size=len(x)))
    
    mles = []
    for n in ns:
        # Construct the initial kernel
        kernel = construct_RBF_kernel(initial_params)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=num_restarts, alpha=eps)
        
        # Select a random subset of x and y
        subset_indices = np.random.choice(n_max, size=n, replace=False)
        subset_x = x[subset_indices]
        subset_y = y[subset_indices]
        
        # Fit the GP model
        gp.fit(subset_x, subset_y)
        
        # Extract the fitted kernel parameters
        gp_params = gp.kernel_.get_params()
        param_estimates = [gp_params['k1__constant_value'], gp_params['k2__length_scale']]
        
        # Append the estimated parameters to mles
        mles.append(param_estimates)
    
    # Return the estimated parameters
    return mles

# Parameters
n_max = 500
ns = [500]
true_params = [1, 1 / 500]
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


# In[60]:


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
    np.save('./results/RBF-simulation.npy', param_estimates)
    return param_estimates

# In[75]:


# Timing execution
start_time = time.time()

# Parameters
num_replicates = 100
true_params = [1, 1 / 500]  # Example true parameters
initial_params = [1 * param for param in true_params]  # Example initial parameters
eps = 0.1
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




