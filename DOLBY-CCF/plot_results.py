import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from astropy.io import fits

# Function to plot Gaussian fit results
def plot_gaussian_fit(velocity, ccfs, ccfs_err, popt):
    from gaussian_fit import custom_model  # Move import inside the function
    
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.errorbar(velocity, ccfs, yerr=ccfs_err, fmt='.', zorder=0)
    ax1.plot(velocity, custom_model(np.array(velocity), *popt), 'r-', label='Gaussian fit', zorder=1)
    ax1.set_xlabel('Velocity (km/s)')
    ax1.set_ylabel('Cross-correlation')
    ax1.set_title('CCF Gaussian Fit')
    plt.legend()
    plt.savefig(f'gaussian_fit.png', dpi=100, bbox_inches='tight')

# Function to plot Gaussian Process fit results
def plot_gp_fit(velocity, ccfs, ccfs_err, t, mu, std):
    plt.figure(figsize=(10, 6))
    plt.errorbar(velocity, ccfs, yerr=ccfs_err, fmt=".k", capsize=0, alpha=0.5, label='Data')
    plt.plot(t, mu, color='red', label='GP Prediction')
    plt.fill_between(t, mu + std, mu - std, color='red', alpha=0.3)
    plt.xlabel("Velocity (km/s)")
    plt.ylabel("Cross-correlation")
    plt.title("Gaussian Process Fit")
    plt.legend()
    plt.savefig(f'gp_fit.png', dpi=100, bbox_inches='tight')


# Function to plot MCMC results
def plot_mcmc_results(sampler):
    # Define MCMC parameters
    burn_steps = 2500  # Number of steps to discard before sampling
    thin_by = 50  # Step interval to thin the chain by
    discard = int(1. * burn_steps / thin_by)  # Number of discarded steps
    samples = sampler.get_chain(flat=True, discard=discard)  # Get the flattened chain
    names = [r'gp: $\log{\sigma1}$', r'gp: $\log{\rho}$', r'gp: $\log{\sigma2}$', r'gp: $\log{\rho}$',
             "Contrast1", "RV1", "FWHM1", "Contrast2", "RV2", "FWHM2"]  # Parameter names
    plt.clf()
    fig = corner.corner(samples, labels=names, show_titles=True,
                        title_kwargs={"fontsize": 12}, quantiles=[0.16, 0.5, 0.84])  # Plot corner plot
    fig.savefig('mcmc_corner.png', dpi=250, bbox_inches='tight')  # Save plot to file

if __name__ == "__main__":
    # Load the results from the Gaussian fit
    results_file = 'gaussian_results.txt'
    results_df = pd.read_csv(results_file, sep='\t')

    # Example index for the CCF file to analyze
    index = 0  # Change this to the desired index
