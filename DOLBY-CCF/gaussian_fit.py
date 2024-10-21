import numpy as np
from scipy.optimize import curve_fit
from glob import glob
from plot_results import plot_gaussian_fit
# Define the bimodal function with two Gaussian components
def bimodal_ccf(x, offset1, amp1, mu1, std1, offset2, amp2, mu2, std2):
    y0 = 1. - offset1 - amp1 * np.exp(-(x - mu1) ** 2 / (2. * std1 ** 2))
    y1 = 1. - offset2 - amp2 * np.exp(-(x - mu2) ** 2 / (2. * std2 ** 2))
    y_total = (y0 + y1 - 1.)
    return y_total

# Define the linear function
def linear(x, A, B):
    return A + B * x

# Define the custom model that combines the bimodal and linear functions
def custom_model(x, offset1, amp1, mu1, std1, offset2, amp2, mu2, std2, A, B):
    return bimodal_ccf(x, offset1, amp1, mu1, std1, offset2, amp2, mu2, std2) + linear(x, A, B)

# Load the CCF files
ccf_files = sorted(glob('*ccf*.txt'))

# Open the results file for writing Gaussian fit results
with open('gaussian_results.txt', 'w') as f:
    # Write header line
    header = ['i', 'filename', 'red_chi2', 'mu1', 'mu2', 'contrast1', 'contrast2', 'fwhm1', 'fwhm2']
    f.write('\t'.join(header) + '\n')

    # Loop through CCF files for Gaussian fitting
    for i in range(len(ccf_files)):
        velocity = []
        ccfs = []

        # Read CCF file
        with open(ccf_files[i], 'r') as file:
            for line in file:
                columns = line.split()
                velocity.append(float(columns[0]))
                ccfs.append(float(columns[1]))

        # Set initial guess values for fitting
        yy = [1.0 - y for y in ccfs]
        amp1 = np.max(yy)
        mu1 = velocity[np.argmax(yy)]
        std1 = 10.0
        offset1 = 0.
        offset2 = 0.
        amp2 = amp1
        mu2 = mu1 - 100 if mu1 >= 0 else mu1 + 100
        mu2 = max(mu2, 0)
        std2 = 5.0
        A = np.median(ccfs)
        B = 0.0

        # Calculate uncertainties for each value of CCF
        cont_region = (np.array(velocity) > 150) & (np.array(velocity) < 180)
        noise_std = np.std(np.array(ccfs)[cont_region])
        ccfs_err = np.ones_like(ccfs) * noise_std

        # Fit the model to the data
        popt, pcov = curve_fit(custom_model, velocity, ccfs, sigma=ccfs_err,
                               p0=[offset1, amp1, mu1, std1, offset2, amp2, mu2, std2, A, B],
                               bounds=([-1, -1, -200, 0.1, -1, -1, -200, 0.1, np.min(ccfs), -10],
                                       [1, 1, 200, 100, 1, 1, 200, 100, np.max(ccfs), 10]))

        perr = np.sqrt(np.diag(pcov))

        # Calculate reduced chi-squared
        residuals = ccfs - custom_model(np.array(velocity), *popt)
        chi2 = np.sum((residuals / ccfs_err) ** 2)
        dof = len(velocity) - len(popt)
        red_chi2 = chi2 / dof

        # Save results to file
        contrast1 = 100. * popt[1]
        rv1 = popt[2]
        fwhm1 = 2.35482004503 * popt[3]
        rv2 = popt[6]
        contrast2 = 100. * popt[5]
        fwhm2 = 2.35482004503 * popt[7]
        results = [str(i), ccf_files[i], str(red_chi2), str(rv1), str(rv2),
                   str(contrast1), str(contrast2), str(fwhm1), str(fwhm2)]
        f.write('\t'.join(results) + '\n')

        # Call the plotting function for the Gaussian fit
        plot_gaussian_fit(velocity, ccfs, ccfs_err, popt)

# Close the results file
f.close()
