# DOLBY-CCF
DOuble-Lined BinarY -- Cross Correlation Function analysis code using Gaussian process

This repository contains code for analysing Cross-Correlation Functions of double-line binary stars using Gaussian Process modeling. 

The scripts included are:

- 'DOLBY_ccf.py`: The main script that orchestrates the analysis, reading CCF data, fitting Gaussian profiles, performing Gaussian Process modeling, and generating plots.
- `gaussian_fit.py`: Fits Gaussian profiles to the CCF data.
- `gp_fit.py`: Performs Gaussian Process regression on the fitted Gaussian profiles.
- `plot_results.py`: Contains plotting functions to visualise the results.

## Installation

To run the code, you will need to install the required Python packages. You can do this using `pip`:

pip install -r requirements.txt

## Citing

If you use DOLBY-CCF in your research, please cite:

**Lalitha Sairam et al. 2023**: *New methods for radial-velocity measurements of double-lined binaries, and detection of a circumbinary planet orbiting TIC 172900988*. Monthly Notices of the Royal Astronomical Society, 527(2), 2261-2278. [https://doi.org/10.1093/mnras/stad3136](https://doi.org/10.1093/mnras/stad3136)

or use the ready-made BibTeX entry:

```bibtex
@article{Lalitha_Sairam_2023,
    author = {Sairam, Lalitha and Triaud, Amaury H M J and Baycroft, Thomas A and Orosz, Jerome and Boisse, Isabelle and Heidari, Neda and Sebastian, Daniel and Dransfield, Georgina and Martin, David V and Santerne, Alexandre and Standing, Matthew R},
    title = "{New methods for radial-velocity measurements of double-lined binaries, and detection of a circumbinary planet orbiting TIC 172900988}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {527},
    number = {2},
    pages = {2261-2278},
    year = {2023},
    month = {11},
    abstract = "{Ongoing ground-based radial-velocity observations seeking to detect circumbinary planets focus on single-lined binaries even though over 9 in every 10 binary systems in the solar neighbourhood are double lined. Double-lined binaries are on average brighter, and should in principle yield more precise radial velocities. However, as the two stars orbit one another, they produce a time-varying blending of their weak spectral lines. This makes an accurate measure of radial velocities difficult, producing a typical scatter of \$10\{\\!-\\!\}15~\\rm m\\, s^\{-1\}\$. This extra noise prevents the detection of most orbiting circumbinary planets. We develop two new data-driven approaches to disentangle the two stellar components of a double-lined binary, and extract accurate and precise radial velocities. Both approaches use a Gaussian process regression, with the first one working in the spectral domain, whereas the second works on cross-correlated spectra. We apply our new methods to TIC 172900988, a proposed circumbinary system with a double-lined binary, and detect a circumbinary planet with an orbital period of \$150~\\rm d\$, different than previously proposed. We also measure a significant residual scatter, which we speculate is caused by stellar activity. We show that our two data-driven methods outperform the traditionally used TODCOR and TODMOR, for that particular binary system.}",
    issn = {0035-8711},
    doi = {10.1093/mnras/stad3136},
    url = {https://doi.org/10.1093/mnras/stad3136},
    eprint = {https://academic.oup.com/mnras/article-pdf/527/2/2261/53404012/stad3136.pdf},
}

