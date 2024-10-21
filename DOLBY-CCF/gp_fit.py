import numpy as np
import pandas as pd
from scipy.optimize import minimize
import celerite
from celerite import terms
import emcee
import corner
import matplotlib.pyplot as plt
from astropy.io import fits as fits
from plot_results import plot_gp_fit, plot_mcmc_results  # Import your plotting functions

# Load the results from the Gaussian fit
results_file = 'gaussian_results.txt'
results_df = pd.read_csv(results_file, sep='\t')

# Example index for the CCF file to analyze
index = 0  # Change this to the desired index

# Extract the relevant parameters for the specified index
row = results_df.iloc[index]
filename = row['filename']
rv1 = float(row['mu1'])
rv2 = float(row['mu2'])
contrast1 = float(row['contrast1'])
contrast2 = float(row['contrast2'])
fwhm1 = float(row['fwhm1'])
fwhm2 = float(row['fwhm2'])

# Load the CCF data
velocity, ccfs, ccfs_err = [], [], []

with open(filename, 'r') as file:
    for line in file:
        columns = line.split()
        velocity.append(float(columns[0]))
        ccfs.append(float(columns[1]))

# Convert lists to numpy arrays
velocity = np.array(velocity)
ccfs = np.array(ccfs)
cont_region = (np.array(velocity) > 150) & (np.array(velocity) < 180)
noise_std = np.std(np.array(ccfs)[cont_region])
ccfs_err = np.ones_like(ccfs) * noise_std

# Create the GP model
# Define the celerite model for the GP
class Model_celerite(celerite.modeling.Model):
    parameter_names = ("Contrast1", "RV1", "FWHM1", "Contrast2", "RV2", "FWHM2")
    def get_value(self, x):
        # Calculate the two Gaussian components using the input parameters
        amp1 = self.Contrast1 / 100
        mu1 = self.RV1
        std1 = self.FWHM1 / 2.35482004503
        amp2 = self.Contrast2 / 100
        mu2 = self.RV2
        std2 = self.FWHM2 / 2.35482004503
        return 1. - amp1 * np.exp(-0.5 * ((x.flatten() - mu1) / std1)**2) - amp2 * np.exp(-0.5 * ((x.flatten() - mu2) / std2)**2)

# Define initial guesses and bounds for GP parameters
log_sigma_init1 = np.log(1)  # Adjust as necessary
log_sigma_init2 = np.log(1)  # Adjust as necessary
log_rho_init = np.log(np.mean(np.abs(np.diff(velocity))))  # Adjust as necessary

# Initialize the GP model parameters
mean_model = Model_celerite(Contrast1=contrast1, RV1=rv1, FWHM1=fwhm1, Contrast2=contrast2, RV2=rv2, FWHM2=fwhm2)

# Set bounds for the hyperparameters
bounds = dict(log_sigma=(-15, 15), log_rho=(-10, 15))

# Create two Matern32 kernels, one for each Gaussian component
kernel1 = terms.Matern32Term(log_sigma=log_sigma_init1, log_rho=log_rho_init, bounds=bounds)
kernel2 = terms.Matern32Term(log_sigma=log_sigma_init2, log_rho=log_rho_init, bounds=bounds)

# Combine the two kernels into a single kernel
kernel = kernel1 + kernel2

# Create celerite GP model with specified kernel and mean model
gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)

# Compute the GP model with input data and uncertainties
gp.compute(velocity, yerr=ccfs_err)

# Print initial log-likelihood
print("Initial log-likelihood: {0}".format(gp.log_likelihood(ccfs)))

# Define negative log-likelihood function for optimization
def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

# Get initial parameters for optimization
initial_params = gp.get_parameter_vector()
ndim = len(initial_params)
bounds = gp.get_parameter_bounds()

# Optimize the GP model using L-BFGS-B method with bounds on parameters
soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(ccfs, gp))
gp.set_parameter_vector(soln.x)

# Print final log-likelihood after optimization
print("Final log-likelihood: {0}".format(-soln.fun))

# Predict mean and variance of GP model
t = np.linspace(np.min(velocity), np.max(velocity), 100)  # Adjust the number of points as needed
mu, var = gp.predict(ccfs, t, return_var=True)
std = np.sqrt(var)

# Plot the GP model along with the data
plot_gp_fit(velocity, ccfs, ccfs_err, t, mu, std)

# Save the GP model and data if needed
np.save(f"gp_model_{index}.npy", np.vstack((t, mu, std)))

# MCMC
def log_probability(params, x, y):
    gp.set_parameter_vector(params)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    return gp.log_likelihood(y) + lp

# Initial values and dimensions for MCMC
initial = np.array(soln.x)
ndim, nwalkers = len(initial), 50
p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(velocity, ccfs))

# Burn-in phase
p0, lp, _ = sampler.run_mcmc(p0, 500)
sampler.reset()

# Production phase
p = p0[np.argmax(lp)]
p0 = [np.array(p) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
nsteps = 5000
p0, _, _ = sampler.run_mcmc(p0, nsteps)

# Corner plot
# Define MCMC parameters
burn_steps = 2500 # number of steps to discard before sampling
thin_by = 50 # step interval to thin the chain by
discard = int(1. * burn_steps / thin_by) # number of discarded steps
samples = sampler.get_chain(flat=True, discard=discard) # get the flattened chain
names = [r'gp: $\log{\sigma1}$', r'gp: $\log{\rho}$', r'gp: $\log{\sigma2}$', r'gp: $\log{\rho}$', "Contrast1", "RV1", "FWHM1", "Contrast2", "RV2", "FWHM2"] 

# Call the plotting function for the MCMC results
plot_mcmc_results(samples)  # Call your MCMC plotting function

# Calculate percentiles for parameters of interest
rvgp1_16, rvgp1_50, rvgp1_84 = np.percentile(samples[:, 5], [16, 50, 84])
rvgp2_16, rvgp2_50, rvgp2_84 = np.percentile(samples[:, 8], [16, 50, 84])
congp1_16, congp1_50, congp1_84 = np.percentile(samples[:, 4], [16, 50, 84])
congp2_16, congp2_50, congp2_84 = np.percentile(samples[:, 7], [16, 50, 84])
fwhmgp1_16, fwhmgp1_50, fwhmgp1_84 = np.percentile(samples[:, 6], [16, 50, 84])
fwhmgp2_16, fwhmgp2_50, fwhmgp2_84 = np.percentile(samples[:, 9], [16, 50, 84])

# Save results to file
with open('gp_results.txt', 'a') as f:
    red_chi2 = -soln.fun  # Calculate your reduced chi-squared value as needed
    perr = np.std(samples, axis=0)  # Calculate parameter errors (you may adjust this as necessary)
    
    results = [
        str(index), filename, str(red_chi2), str(rv1), '+/-', str(perr[2]),
        str(rv2), '+/-', str(perr[6]), str(contrast1), '+/-', str(100 * perr[1]),
        str(contrast2), '+/-', str(100 * perr[5]), str(fwhm1), '+/-', str(2.35482004503 * perr[3]),
        str(fwhm2), '+/-', str(2.35482004503 * perr[7]),
        str(rvgp1_50) + ' ' + str(rvgp1_84 - rvgp1_50) + ' ' + str(rvgp1_50 - rvgp1_16),
        str(rvgp2_50) + ' ' + str(rvgp2_84 - rvgp2_50) + ' ' + str(rvgp2_50 - rvgp2_16),
        str(congp1_50) + ' ' + str(congp1_84 - congp1_50) + ' ' + str(congp1_50 - congp1_16),
        str(congp2_50) + ' ' + str(congp2_84 - congp2_50) + ' ' + str(congp2_50 - congp2_16),
        str(fwhmgp1_50) + ' ' + str(fwhmgp1_84 - fwhmgp1_50) + ' ' + str(fwhmgp1_50 - fwhmgp1_16),
        str(fwhmgp2_50) + ' ' + str(fwhmgp2_84 - fwhmgp2_50) + ' ' + str(fwhmgp2_50 - fwhmgp2_16),
    ]
    f.write('\t'.join(results) + '\n')
