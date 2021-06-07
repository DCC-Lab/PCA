#  Understanding Principal Components Analysis for spectral analysis



[TOC]

## Preamble

The goal of this at this point is to understand Principal Components Analysis (PCA) in scikit-learn.decomposition.

I have done this by writing a sequence of tests that helped me understand what was going on.

This document is a work in progress and sections will improve with my understanding.



## Introduction

Spectoscopy is the optical method of choice to detect substances or identify tissues.  We have learned from very early on that identifying the peaks in a spectrum can be done to infer the substances in our sample. For instance, if we have pure substances, it is relatively easy to identify Ethanol and Methanol with their Raman spectra, because their shapes are significantly different and many peaks do not overlap:

<img src="README.assets/csm_fig_3_Raman_spectra_of_ethanol_and_methanol_dabf593771.png" alt="What is Raman Spectroscopy? - HORIBA" style="zoom:48%;" />

It may even be possible to separate both substances if we have a mixture of the two, by fitting $c_e S_(\nu)_{e} + c_m S(\nu)_{m}$ to find the appropriate concentrations that can explain our final combined spectrum. However, what do we do if we have a mixture of several solutions? What if several peaks overlap? What if we don't know the original spectra? 

We know intuitively that if peaks belong to the same molecule, they should vary together.  If by chance none of the peaks from the different analytes overlap, then it becomes trivial: we only need to identify the peaks, find their amplitudes, and we will quickly get the concentrations of the respective analytes. But things get complicated if they have overlapping peaks, and even worse if we have more than a few components.  

From a mathematical point of view, we can consider a spectrum as a **vector**:
$$
\mathbf{S} = \sum_{i=0}^{N} I_i\mathbf{\hat{\nu}}_i
$$
where each individual frequency $\nu_i$ is in its own dimension, with $\hat{\nu}_i$ the base vector.  Therefore, if we have 1024 points in our intensity spectrum, we are in an N-dimensional space. 

