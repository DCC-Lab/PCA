#  Understanding Principal Components Analysis for spectral analysis



[TOC]

## Preamble

The goal of this at this point is to understand Principal Components Analysis (PCA) in scikit-learn.decomposition.

I have done this by writing a sequence of tests that helped me understand what was going on.

This document is a work in progress and sections will improve with my understanding.



## Note

Use [Typora](http://typora.io) to read this document to see the equations.

## Introduction

Spectroscopy is the optical method of choice to detect substances or identify tissues.  We have learned from very early on that identifying the peaks in a spectrum can be done to infer the substances in our sample. For instance, if we have pure substances, it is relatively easy to identify Ethanol and Methanol with their Raman spectra, because their shapes are significantly different and many peaks do not overlap:

<img src="README.assets/csm_fig_3_Raman_spectra_of_ethanol_and_methanol_dabf593771.png" alt="What is Raman Spectroscopy? - HORIBA" style="zoom:48%;" />

It may even be possible to separate both substances if we have a mixture of the two, by fitting $c_e S_(\nu)_{e} + c_m S(\nu)_{m}$ to find the appropriate concentrations that can explain our final combined spectrum. However, what do we do if we have a mixture of several solutions? What if several peaks overlap? What if we don't know the original spectra? 

We know intuitively that if peaks belong to the same molecule, they should vary together.  If by chance none of the peaks from the different analytes overlap, then it becomes trivial: we only need to identify the peaks, find their amplitudes, and we will quickly get the concentrations of the respective analytes. But things get complicated if they have overlapping peaks, and even worse if we have more than a few components.  

## Setting things up: spectra as vectors


From a mathematical point of view, we can consider a spectrum as a **vector** of intensities:
$$
\mathbf{S} = \sum_{i=1}^{N} I_i\mathbf{\hat{\nu}}_i,
$$
where each individual frequency $\nu_i$ is in its own dimension, with $\hat{\nu}_i$ the base vectors and $I_i$ is the intensity at that frequency.  Therefore, if we have 1024 points in our intensity spectrum, we are in an N-dimensional space, with the components being $(I_1,I_2,...I_N)$, and we should assume (at least for now) that these components are all independent. If we define the norm of a vector from the dot product, we can say that the norm is equal to:
$$
\left|\mathbf{S} \right|^2 = \sum_{i=1}^{N} I_i\mathbf{\hat{\nu}}_i \cdot \sum_{j=1}^{N} I_j\mathbf{\hat{\nu}}_j = \sum_{i=1}^{N}\sum_{j=1}^{N} I_i I_j\ \hat{\nu}_i \cdot \hat{\nu}_j = \sum_{i=1}^{N} \left|I_i\right|^2
$$
since the spectral base vectors $\hat{\nu}_i$ are all orthonormal, which we can use to normalize a spectrum (or a vector). Finally, it is very convenient to work with matrix notation to express many of these things.  We can express the spectrum $\mathbf{S}$ in the basis $\left\{ \hat{\nu}_i \right\}$ with:
$$
\mathbf{S} = \sum_{i=1}^{N} I_i\mathbf{\hat{\nu}}_i 
=
\left( \mathbf{\hat{\nu}}_1 \ \mathbf{\hat{\nu}}_2\  ...  \mathbf{\hat{\nu}}_N \right)
\left( I_1 \ I_2 \ ... \ I_N \right)^T
=
\left( \mathbf{\hat{\nu}}_1 \ \mathbf{\hat{\nu}}_2\  ...  \mathbf{\hat{\nu}}_N \right)
\left( 
\begin{matrix}
I_1 \\
I_2 \\
... \\
I_N \\
\end{matrix}
\right)
$$
If we consider these matrices as partitions, we can write in a form even more compact as:
$$
\mathbf{S} = \hat{\nu} \cdot [\mathbf{I}]_\nu
$$
We will use the transpose notation to keep expressions on a single line.



## Spectra as non-independent values

However, we know from experience that in a spectrum, intensities are not completely independent: for instance, in the methanol spectrum above, the peak around 1000 cm$^{-1}$ has a certain width and therefore those intensities are related and are not independent. In fact, for the spectrum of a single substance, *all intensities* are related because they will come from a scaled version of the original spectrum. Therefore, if we have the reference methanol spectrum:
$$
\mathbf{\hat{B}}_M = \sum_{i=0}^{N} I_{M,i}\mathbf{\hat{\nu}}_i,
$$
any other solution of methanol of concentration $c_M$ would simply yield the spectrum:
$$
\mathbf{S} = c_M\mathbf{\hat{B}}_M = c_M \sum_{i=0}^{N} I_{M,i}\mathbf{\hat{\nu}}_i.
$$
So if we have several base solutions from which we create a large number of samples, we can write:
$$
\mathbf{S} = \sum_j c_j\mathbf{\hat{B}}_j = \mathbf{\hat{B}} \cdot [ c ]_\mathbf{B}
$$
And if we want to describe the $M$ spectra from samples otbained from mixing of these base solutions $\mathbf{B}$ with concentrations $C_\mathbf{B}$ we can write:
$$
\mathbf{S} = \mathbf{\hat{B}} \cdot C_\mathbf{B}
$$



