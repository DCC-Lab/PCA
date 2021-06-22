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

## Spectra as vectors


From a mathematical point of view, we can consider a spectrum as a **vector** of intensities:
$$
\mathbf{I} = \sum_{i=1}^{n} I_i\mathbf{\hat{\nu}}_i,
$$
where each individual frequency $\nu_i$ is in its own dimension, with $\hat{\nu}_i$ the base vectors and $I_i$ is the intensity at that frequency.  Therefore, if we have N=1024 points in our intensity spectrum $\mathbf{I}$, we are in an N-dimensional space, with the components being $(I_1,I_2,...I_N)$, and we should assume (at least for now) that these components are all independent. If we define the **norm** of a vector from the **dot product**, we can say that the norm is equal to:
$$
\left|\mathbf{I} \right|^2 = \sum_{i=1}^{n} I_i\mathbf{\hat{\nu}}_i \cdot \sum_{j=1}^{n} I_j\mathbf{\hat{\nu}}_j = \sum_{i=1}^{n}\sum_{j=1}^{n} I_i I_j\ \hat{\nu}_i \cdot \hat{\nu}_j = \sum_{i=1}^{n} \left|I_i\right|^2,
$$
since the spectral base vectors $\hat{\nu}_i$ are all orthonormal, which we can use to normalize a spectrum (or a vector). Finally, it is very convenient to work with matrix notation to express many of these things.  We can express the spectrum $\mathbf{I}$ in the basis $\left\{ \hat{\nu}_i \right\}$ with:
$$
\mathbf{I} = \sum_{i=1}^{n} I_i\mathbf{\hat{\nu}}_i 
=
\left( \mathbf{\hat{\nu}}_1 \ \mathbf{\hat{\nu}}_2\  ...  \mathbf{\hat{\nu}}_n \right)
\left( I_1 \ I_2 \ ... \ I_n \right)^T
=
\left( \mathbf{\hat{\nu}}_1 \ \mathbf{\hat{\nu}}_2\  ...  \mathbf{\hat{\nu}}_n \right)
\left( 
\begin{matrix}
I_1 \\
I_2 \\
... \\
I_n \\
\end{matrix}
\right)
$$
If we consider these matrices as partitions, we can write in a form even more compact as:
$$
\mathbf{I} = \hat{\nu} [I]_\nu
$$
where the notation $\left[ I\right]_\nu$ means "the intensity coefficients in base $\nu$  to multiply the base vectors $\hat{\nu}$ and obtain the vector (spectrum)". We will use the transpose notation to keep expressions on a single line when needed.

Note that the vector itself $\mathbf{I}$ is different from the *components of that vector in a given basis* $[I]_\nu$.  For more information about the notation for vector, base vectors and coefficients:

* Read [Greenberg Section 10.7](./Greenberg base change.pdf) on bases and base changes.
* Watch [the video](https://www.youtube.com/watch?v=FNuKax5NEpw&list=PLUxTghemi4FvGibCevLK8S89Q7d_eC9HX&index=33) (in French) that explains in even more details where this comes from.
* Watch [an example](https://www.youtube.com/watch?v=REWGdCBoAxI&list=PLUxTghemi4FvGibCevLK8S89Q7d_eC9HX&index=32) (in French) for problem 10.7.1 that discussed an application of this notation and formalism to perform a base change.

## Bases

A reminder for the definition of a base $\left\{ \mathbf{e}_i \right\}$:

1. A base set is **complete**: it spans the space for which it is a base: you must be able to get every vector in that space with $\mathbf{v} = \sum c_i \mathbf{e}_i$. We call the $c_i$ the components of a vector *in that base*.
2. A base set is **linearly independent**: all base vectors are independent, and the only way to combine the base vectors to obtain the null vector $\sum c_i \mathbf{e}_i = \mathbf{0}$ is with $c_i =0$ for all $c_i$. 
3. The number of base vectors in the set is the dimension of the space.

Notice that :

1. The base vectors **do not have to be unitary**: they can have any length. A **normalized** base set will be labelled with a hat on the vector $\left\{ \mathbf{\hat{e}}_i \right\}$, and an arbitrary set will be $\left\{ \mathbf{e}_i \right\}$.
2. The base vectors **do not have to be orthogonal**: as long as they are independent, that is fine. There is no notation to differentiate orthogonal and non-orthogonal basis because the property of orthogonality is not a single vector property, it is a property of a pair of vectors, therefore we cannot label a vector as "orthogonal".

## Spectra as *dependent* vectors

However, we know from experience that in a spectrum, intensities are not completely independent: for instance, in the methanol spectrum above, the peak around 1000 cm$^{-1}$ has a certain width and therefore those intensities are related and are not independent. In fact, for the spectrum of a single substance, *all intensities* are related because they will come from a scaled version of the original spectrum. Therefore, if we have the reference methanol spectrum for a unity concentration $\mathbf{\hat{s}}_M$:
$$
\mathbf{\hat{s}}_M = \sum_{i=0}^{n} I_{M,i}\mathbf{\hat{\nu}}_i,
$$
where $I_{M,i}$ is the relative intensity at frequency $\nu_i$. Any other solution of methanol of scalar concentration $c_M$ would simply yield the spectrum:
$$
\mathbf{I} = c_M\mathbf{\hat{s}}_M = c_M \sum_{i=0}^{n} I_{M,i}\mathbf{\hat{\nu}}_i.
$$
So if we have several individual solutions $\left\{\mathbf{\hat{s}}_j\right\}$ from which we create a mixture of concentrations $c_j$, we will generate spectra in a sub-space of the original $n$-dimensional intensity vector-space. The set of vectors $\left\{\mathbf{\hat{s}}_j\right\}$ is a basis set because we can generate all vectors in that sub-space with a linear combination of the vectors (or spectra). The dimension of that subspace is equal to the number of elements in $\left\{\mathbf{\hat{s}}_j\right\}$ We can write the mixture spectrum $\mathbf{I}$ as:
$$
\mathbf{I} = \sum_j c_j\mathbf{\hat{s}}_j = \left( \mathbf{\hat{s}}_1, \mathbf{\hat{s}}_2,...,\mathbf{\hat{s}}_n \right)  \left( c_1, c_2,...,c_n \right)^T = \mathbf{\hat{s}}  [ c ]_\mathbf{\hat{s}}
$$
Again, we read the last expression $[ c ]_\mathbf{\hat{s}}$ as "the coefficients in base $\left\{\mathbf{\hat{s}}\right\}$ needed to multiply the base vectors $\mathbf{\hat{s}}_i$ to obtain the final spectrum $\mathbf{I}$". It stresses the point that the vector $\mathbf{I}$ and its components in a given basis $[ c ]_\mathbf{\hat{s}}$ are not the same thing. This will become critical below when we look a Principal Components.

Finally, if we want to describe a **collection** of $m$ spectra obtained from mixing these base solutions $\mathbf{\hat{s}}$ with concentrations $c_{ij}$ for the i-th spectrum and the j-th base solution, we can write:

$$
\mathbf{I}_i = \sum_{j} c_{ij}\mathbf{\hat{s}}_j.
$$

This can be rewritten in matrix notation:
$$
\left( \mathbf{I}_1, \mathbf{I}_2,...,\mathbf{I}_m \right) = \left( \mathbf{\hat{s}}_1, \mathbf{\hat{s}}_2,...,\mathbf{\hat{s}}_n \right) 
\left( 
\begin{matrix}
c_{11} & c_{21} & ... & c_{m1} \\
c_{12} & c_{22} & ... & c_{m2} \\
... & ... & ...& ...\\
c_{1n} & c_{2n} & ... & c_{mn}
\end{matrix}
\right)
$$

to yield this very compact form:
$$
\mathbf{I} = \mathbf{\hat{s}} \left[\mathbf{C}\right]_\mathbf{\hat{s}}
$$

This equation represents, in a single expression, the $m$ spectra obtained by mixing the $n$ solutions with concentrations $c_{ij}$ for the $i$-th spectrum and the $j$-th solution. 

## Final notes on intensity spectra as vectors

If we have several components (i.e. methanol, ethanol, etc...) and there is no overlap whatsoever between the spectra (i.e the peaks are all distinct), then the base vectors $\hat{b}_i$ and $\hat{b}_j$ are orthogonal. However, it is more likely that the solutions *do* have overlapping spectra, therefore the base vectors (and consequently the base itself) will *not be orthogonal*. It is perfectly acceptable to have a base that is not orthogonal: it remains a base because any linear combination can create any spectrum we would measure.

## Base change

The general expression for a vector as a function of its basis and its components in that basis is such that obviously, it stands correct for any basis:
$$
\mathbf{I} = \mathbf{{e}} \left[\mathbf{C}\right]_\mathbf{e} =\mathbf{{e}^\prime} \left[\mathbf{C^\prime}\right]_\mathbf{e^\prime}
\label{eq:vectorBasis}
$$
It is the purpose of the present section to show how to go from a basis b to a basis b', that is, how to transform the coefficients c into coefficients c'. For more information, you can look at the [Youtube Video](https://www.youtube.com/watch?v=FNuKax5NEpw&list=PLUxTghemi4FvGibCevLK8S89Q7d_eC9HX&index=33) on base changes. 

Since we can express any vector in a basis, we can choose to express the basis vectors $\mathbf{{e}}$ in the $\mathbf{{e}}^\prime$ basis, with the coefficients $\left[ \mathbf{Q} \right]_{\mathbf{{e}}^\prime}$we do not know yet:  
$$
\mathbf{{e}} = \mathbf{{e}}^\prime \left[ \mathbf{Q} \right]_{\mathbf{{e}}^\prime},
\label{eq:bprimetob}
$$
where each column of the matrix $\left[ \mathbf{Q} \right]_{\mathbf{{e}}^\prime}$ is the component of the vector $\mathbf{e}_i$ in the $\mathbf{e}^\prime$ basis. By definition, a basis set has enough vectors to cover the vector space, therefore both basis sets must have the same number of vectors, and the matrix $\left[ \mathbf{Q} \right]_{\mathbf{{e}}^\prime}$ is necessarily square, and can be inverted. We can therefore use $(\ref{eq:vectorBasis})$ in $(\ref{eq:bprimetob})$ and obtain simply:
$$
\mathbf{I} = \mathbf{{e}} \left[\mathbf{C}\right]_\mathbf{e} 
=
\left( \mathbf{{e}}^\prime \left[ \mathbf{Q} \right]_{\mathbf{{e}}^\prime} \right)
\left[\mathbf{C}\right]_\mathbf{e}
=
\mathbf{{e}}^\prime \left( \left[ \mathbf{Q} \right]_{\mathbf{{e}}^\prime}
\left[\mathbf{C}\right]_\mathbf{e} \right)
=
\mathbf{{e}^\prime} \left[\mathbf{C^\prime}\right]_\mathbf{e^\prime}
\label{eq:base}
$$
This means that, when the vectors in different bases are expressed by $(\ref{eq:vectorBasis})$, the coordinates in the basis $\mathbf{e}^\prime$ can be obtained from the components in the basis $\mathbf{e}$ by this simple transformation:
$$
\left[\mathbf{C^\prime}\right]_\mathbf{e^\prime} \equiv \left[ \mathbf{Q} \right]_{\mathbf{\hat{e}}^\prime}
\left[\mathbf{C}\right]_\mathbf{e}
$$



## Principal Component Analysis (PCA) in `sklearn`

 The goal of Principal Component Analysis (PCA) is to obtain an orthogonal basis for a much smaller subspace than the original (it is a *dimensionality reduction* technique). We will identify this **orthonormal** PCA base as $\left\{ \mathbf{\hat{p}} \right\}$, known as the principal component basis, or just the principal components. 

At this point, it is farily simple to describe the process without worrying about the details: PCA takes a large number of samples spectra, and will:

1. Find the *principal components*, or an orthonormal basis $\left\{ \mathbf{\hat{p}} \right\}$ that explains the variance of the data the best with a value that expresses how important they are.
2. Given a spectrum $\mathbf{I}$, it can return (*fit*) it to the PCA components and give the coefficients $\left[ \mathbf{I} \right]_\mathbf{\hat{p}}$ in the PCA basis $\left\{ \mathbf{\hat{p}} \right\}$, with $\mathbf{I}=\mathbf{\hat{p}} \left[ \mathbf{I} \right]_\mathbf{\hat{p}}$.

So, because the present document is about the matheatical formalism first and foremost and that I do not want to dive so much into the Python details, let us just say that the following code will give us the principal components, and all the coefficients for our spectra in that base:

```python
from sklearn.decomposition import PCA
#[...]
pca = PCA(n_components=componentsToKeep)
pca.fit(dataSet)                             # find the principal components
# The principal components are available in the variable pca.components_
# They form an orthonormal basis set
pcaDataCoefficients = pca.transform(dataSet) # express our spectra in the PCA basis
# pcaDataCoefficients are the coefficients for each spectrum in the PCA basis
# Note: it is (c1*PC1+c2*PC2+....) + meanSpectrum = Spectrum
# as in equation (15) below.
```



## PCA base change in `sklearn`

Equation $(\ref{eq:vectorBasis})$ is not the only possibility to express a vector in different basis. We will see later that PCA often *translates* the sample vectors (i.e. the intensity spectra) to the "origin" by subtracting the mean spectrum from all spectra. This means that we have a more general transformation than $(\ref{eq:vectorBasis})$ in that we do not express $\mathbf{I}$ in a different set of coordinates but rather $\mathbf{I} - \bar{\mathbf{I}}$:
$$
\mathbf{I}-\mathbf{\bar{I}}= \mathbf{\hat{p}} \left[\mathbf{C}\right]_\mathbf{\hat{p}},
\label{eq:vectorBasis2}
$$
with 
$$
\mathbf{\bar{I}} = \frac{1}{m} \sum_{j=1}^{m} \mathbf{{I}}_{i} \equiv 
\left( \hat{\nu}_1,\hat{\nu}_2, ..., \hat{\nu}_n \right)
\left( \mathbf{\bar{I}}_{1},\mathbf{\bar{I}}_{2}, ..., \mathbf{\bar{I}}_{n} \right)^T
\equiv 
\hat{\nu} \left[ \mathbf{\bar{I}} \right]_\nu
\label{eq:average}
$$
This average is computed in the "intensity" basis (or original basis $\left\{\mathbf{\hat{\nu}}\right\}$) because that is the only basis we know when we start (i.e. we average all spectra). This small change where we subtract the mean is important, because to return to another basis used to generate $\mathbf{I}$, we need to write:
$$
\mathbf{I} 
= 
\mathbf{\hat{p}} \left[\mathbf{C}\right]_\mathbf{\hat{p}} + \mathbf{\bar{I}}
=
\mathbf{{e}} \left[\mathbf{C}\right]_\mathbf{e},
$$
and if we try to follow the same development as in $(\ref{eq:base})$, we would quickly get stuck because we do not have $\mathbf{\bar{I}}$ neither in $\left\{\mathbf{e}\right\}$ or $\left\{\mathbf{\hat{p}}\right\}$ coordinates, we have it in the $\left\{\mathbf{\hat{\nu}}\right\}$ coordinates :
$$
\mathbf{I} 
= 
\mathbf{\hat{p}} \left[\mathbf{C}\right]_\mathbf{\hat{p}} + \hat{\nu} \left[ \mathbf{\bar{I}} \right]_\nu
=
\mathbf{{e}} \left[\mathbf{C}\right]_\mathbf{e},
\label{eq:translated}
$$
Yet, this is the situation we will encounter later:

1. $\mathbf{I} $ is the many spectra we have acquired in the lab.  They are in $\left\{\mathbf{\hat{\nu}}\right\}$ basis (i.e. simple intensity spectra).
2. We can compute $\mathbf{\bar{I}}$ with $(\ref{eq:average})$, also in the spectral component basis $\left\{\mathbf{\hat{\nu}}\right\}$.
3. The $\left\{\mathbf{p}_i\right\}$ basis is the Principal Component Analysis (PCA) basis that will be obtained from the module together with the coefficients $\left[\mathbf{C}\right]_\mathbf{\hat{p}}$.  It comes from a singular value decomposition, and at this point, we do not worry oursleves with how it is obtained: we know we can obtain $\left\{\mathbf{\hat{p}}\right\}$ and $\left[\mathbf{C}\right]_\mathbf{\hat{p}}$ from `sklearn` and PCA.
4. Finally, the $\left\{\mathbf{e}_i\right\}$ basis will be our "solution" basis (or the *physically meaningful* basis)  for which we would like to get the concentrations $\left[\mathbf{C}\right]_\mathbf{e}$ for our lab measurements. In Raman, this could be the lipid spectrum, DNA spectrum, protein spectrum etc...  We know *some* $\left\{\mathbf{e}_i\right\}$ (from insight), but we certainly do not know them all. We want the coefficients to try to determine the concentrations of these molecules and get insight (or answers) from our experimental spectra, but we may not have all the components (i.e. we may not have the full basis set).

There is mathematically not much we can do with these three coordinate systems in $(\ref{eq:translated})$, unless we express the average spectrum $\mathbf{\bar{I}}$ in one or the other bases.  We can do two things:

1. Express $\mathbf{\bar{I}}$ in the base $\left\{\mathbf{e}\right\}$
2. Express $\mathbf{\bar{I}}$ in the PCA base $\left\{\mathbf{\hat{p}}\right\}$

For reasons that should become clear later, we will choose to express $\mathbf{\bar{I}}$ in the base $\left\{\mathbf{\hat{p}}\right\}$ because, in fact, we do not know $\left\{\mathbf{e}\right\}$ completely, we only know *part* of it.  If we knew $\mathbf{{\hat{p}}} \left[ \mathbf{\bar{I}} \right]_\mathbf{\hat{p}}$, we could write:
$$
\mathbf{\hat{p}} \left[\mathbf{C}\right]_\mathbf{\hat{p}} + \mathbf{\hat{p}} \left[ \mathbf{\bar{I}} \right]_\mathbf{\hat{p}}
=
\mathbf{e} \left[\mathbf{C}\right]_\mathbf{e},
$$

$$
\mathbf{\hat{p}} \left( \left[\mathbf{C}\right]_\mathbf{\hat{p}} + \left[ \mathbf{\bar{I}} \right]_\mathbf{\hat{p}} \right)
=
\mathbf{e} \left[\mathbf{C}\right]_\mathbf{e},
$$

If we define for clarity:
$$
\left[\mathbf{C_+} \right]_\mathbf{\hat{p}} \equiv \left[\mathbf{C} \right]_\mathbf{\hat{p}} +  \left[ \mathbf{\bar{I}} \right]_\mathbf{\hat{p}},
$$
we can write:
$$
\mathbf{\hat{p}} \left[\mathbf{C_+} \right]_\mathbf{\hat{p}}
=
\mathbf{{e}} \left[\mathbf{C}\right]_\mathbf{e}.
$$
We obtain it by transforming the null spectrum $\mathbf{0}$ in equation :
$$
\mathbf{0} 
= 
\mathbf{\hat{p}} \left[\mathbf{C_0}\right]_\mathbf{\hat{p}} +  \mathbf{\bar{I}}
$$

$$
\mathbf{\bar{I}} = -\mathbf{\hat{p}} \left[\mathbf{C_0}\right]_\mathbf{\hat{p}} = \mathbf{\hat{p}} \left( - \left[\mathbf{C_0}\right]_\mathbf{\hat{p}}\right) \equiv \mathbf{\hat{p}} \left[ \mathbf{\bar{I}} \right]_\mathbf{\hat{p}}
$$

