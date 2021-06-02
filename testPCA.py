import unittest
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# I will use these later, it is easier if I put them here:
A1 = 1.0 # amplitude of peak1
X1 = 250 # center
W1 = 50  # gaussian width

A2 = 1.0 # same for peak 2
X2 = 600
W2 = 50

skipPlots = False # It is annoying to have to press 'q' to dismiss the plots
                  # if you set this to false, then any test with plots will not run.

"""
I want to understand PCA for spectral analysis. I expect to be able to use
PCA (principal components analysis) to find a "spectral basis" that I can use
to reduce the dimensionality : if, for instance, my spectrum is a combination
of 5 analytes with different concentrations, I expect that PCA will return 5
spectral components that explain most of the variance in my data. These
components will probably not be the spectra of my analytes: they may have 
positive or negative values which makes no physical sense but makes perfect 
mathematical sense.  Therefore I will have to find a way to transform the 
coefficients I get from the PCA module into actual concentrations of analytes
by expressing my analyte spectra in the PCA components found.  Then, I can read
off the concentrations directly.

So that is my goal, but I have never done that ever, and I don't know sklearn.
I will learn sklearn with unittests to validate that I understand. It should
be simple: sklearn.decomposition.pca does all the work. But how do *I* use
it? I will make up "fake" analytes and then build "experimental datasets" with 
noise and see how PCA handles them.
"""


class TestPCA(unittest.TestCase):
    X = None
    C1 = None
    C2 = None

    def testPCAIsImportingProperly(self):
        # Is my module installed properly at least?
        pca = PCA()
        self.assertIsNotNone(pca)

    def testSimulatedXComponent(self):
        # I will need spectra, so let me get started to make sure I can at least do that.
        # Start with a "wavelength" range.
        X  = np.linspace(0,1000,1001)
        self.assertIsNotNone(X)
        self.assertTrue(len(X) != 0)

    def testSimulatedC1Component(self):
        # Create a fake C1 spectrum
        X  = np.linspace(0,1000,1001)
        C1 = A1*np.exp(-(X-X1)**2/W1)
        self.assertIsNotNone(C1)
        self.assertEqual(len(C1), len(X))
        self.assertTrue(np.mean(C1) > 0)

    def testSimulatedC2Component(self):
        # Create another C2 spectrum
        X  = np.linspace(0,1000,1001)
        C2 = A2*np.exp(-(X-X2)**2/W2)
        self.assertIsNotNone(C2)
        self.assertTrue(np.mean(C2) > 0)

    def setUp(self):
        # The setUp() function is called before every test, so let me just call this and create
        # my spectra all the time because I know they work (I just tested them).

        # Once tested, we set them up every time in setUp it will be simpler.
        self.X  = np.linspace(0,1000,1001)
        self.C1 = A1*np.exp(-(self.X-X1)**2/W1)
        self.C2 = A2*np.exp(-(self.X-X2)**2/W2)

    def testSimulatedC1Max(self):
        # Test that the values are fine.
        index = np.argmax(self.C1)
        self.assertEqual(self.X[index], X1)
        self.assertEqual(self.C1[index], A1)

    def testSimulatedC2Max(self):
        # Test that the values are fine.
        index = np.argmax(self.C2)
        self.assertEqual(self.X[index], X2)
        self.assertEqual(self.C2[index], A2)

    def createDataset(self, N):
        # Ok, I can create a basis of 2 spectra, let me "mix" them in a solution
        # and get the "combined" spectrum as if it was an experiment.
        # Create N random combinations (0..1) of C1 and C2
        dataset = []
        for i in range(N):
            a1 = random.random()
            a2 = random.random()
            vector = a1*self.C1 + a2*self.C2
            dataset.append(vector)

        return np.stack(dataset)

    def testDatasetCreation(self):
        # Test my creation function above.
        N = 100
        dataset = self.createDataset(N=N)
        self.assertTrue(len(dataset) == N)
        for v in dataset:
            self.assertTrue(len(v) == len(self.X))

    def newDatasetWithAdditiveNoise(self, dataset, fraction):
        # Ok, I have spectra, I will add noise to make it real
        noisyDataset = []
        for v in dataset:
            noisyVector = []
            for i, value in enumerate(v):
                noise = fraction * random.random() * np.max(v)
                noisyVector.append(v[i] + noise)
            noisyDataset.append(noisyVector)

        return noisyDataset

    def testNoisyDatasetCreation(self):
        # Let me test my noisy spectra
        N = 100
        dataset = self.createDataset(N=N)
        noisyDataset = self.newDatasetWithAdditiveNoise(dataset, 0.05)

        self.assertTrue(len(noisyDataset) == len(dataset))
        for i in range(len(dataset)):
            v = dataset[i]
            vn = noisyDataset[i]

            for j in range(len(v)):
                self.assertTrue(v[j] != vn[j])

    def testPCAIsImportingProperly(self):
        # I am now ready to test PCA with data. Is it still well-installed?
        # (ah ah)
        pca = PCA()
        self.assertIsNotNone(pca)

    @unittest.skipIf(skipPlots, "Skip plots")
    def testFitPCA(self):
        # I will try to perform PCA on my "spectra" that I simulated.  I expect to recover the concentrations
        # that I used when I created them
        # I am following https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        N = 100
        dataset = self.createDataset(N=N)
        noisyDataset = self.newDatasetWithAdditiveNoise(dataset, 0.1)

        pca = PCA(n_components=2)
        pca.fit(noisyDataset)
        self.assertEqual(pca.n_features_, len(self.X))
        self.assertEqual(pca.n_samples_, N)

        fig, ax = plt.subplots()
        plt.plot(pca.components_.transpose())
        ax.set_title("Keeping only 2 components")
        plt.show()

        pca = PCA(n_components=5)
        pca.fit(noisyDataset)
        self.assertEqual(pca.n_features_, len(self.X))
        self.assertEqual(pca.n_samples_, N)

        fig, ax = plt.subplots()
        plt.plot(pca.components_.transpose())
        ax.set_title("Keeping only 5 components")
        plt.show()

        pca = PCA(n_components=10)
        pca.fit(noisyDataset)
        self.assertEqual(pca.n_features_, len(self.X))
        self.assertEqual(pca.n_samples_, N)

        fig, ax = plt.subplots()
        plt.plot(pca.components_.transpose())
        ax.set_title("Keeping only 10 components")
        plt.show()

    def createComponent(self, x, maxPeaks, maxAmplitude, maxWidth, minWidth):        
        # Ok, things are not as expected.  I will create random spectra with a few peaks and varying widths
        N = random.randint(1, maxPeaks)
        
        intensity = np.zeros(len(x))
        for i in range(N):
            amplitude = random.uniform(0, maxAmplitude)
            width = random.uniform(minWidth, maxWidth)
            center = random.choice(x)
            intensity += amplitude*np.exp(-(x-center)**2/width**2)

        return intensity

    def testCreateSpectrum(self):
        # Is the function I just wrote working?
        component = self.createComponent(self.X, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5)
        self.assertIsNotNone(component)
        self.assertTrue(len(component) == len(self.X))

    def createBasisSet(self, x, N, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5):
        # I will create a basis set of a few random "spectra" that I will use as a basis set to create a data set
        basisSet = []
        for i in range(N):
            component = self.createComponent(x, maxPeaks, maxAmplitude, maxWidth, minWidth)
            self.assertIsNotNone(component)
            self.assertTrue(len(component) == len(x))
            basisSet.append(component)

        return np.array(basisSet)

    def testCreateBaseComponents(self, ):
        # Is the function I just wrote working?
        basisSet = self.createBasisSet(self.X, N=5, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5)
        self.assertTrue(basisSet.shape == (5, len(self.X)))

    def createDatasetFromBasisSet(self, N, basisSet):
        # Alright, I have a basis set but now I want a data set

        # I am a bit confused with the numpy dimensions (i.e. the shape) and indices.  Which way do they go?
        # I think it's this:
        # shape = (# base, #spectral_pts)
        
        m, nPts = basisSet.shape
        dataset = []
        for i in range(N):
            vector = np.zeros(nPts)
            for j in range(m):
                cj = random.random()
                vector += cj*basisSet[j]
            dataset.append(vector)

        return np.stack(dataset)

    def testDatasetCreationFromBasisSet(self):
        # Is this createDatasetFromBasisSet working as expected?
        N = 100
        basisSet = self.createBasisSet(self.X, N=5, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataset = self.createDatasetFromBasisSet(N=N, basisSet=basisSet)
        dataset = self.newDatasetWithAdditiveNoise(dataset, fraction=0.1)

        self.assertTrue(len(dataset) == N)
        for v in dataset:
            self.assertTrue(len(v) == len(self.X))

    @unittest.skipIf(skipPlots, "Skip plots")
    def testFitPCAWithMoreComplexBasisSet(self):
        # Alright, now we are really entering the real stuff:
        # I should be able to use PCA and get "eigenvectors" that should resemble my basis set
        # I am following https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        N = 100
        basisSet = self.createBasisSet(self.X, N=5, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataset = self.createDatasetFromBasisSet(N=N, basisSet=basisSet)
        noisyDataset = self.newDatasetWithAdditiveNoise(dataset, fraction=0.1)

        pca = PCA(n_components=2)
        pca.fit(noisyDataset)
        fig, ax = plt.subplots()
        plt.plot(pca.components_.transpose())
        ax.set_title("Keeping only 2 components")
        plt.show() # This looks nothing like my basis set: some values are negative! I don't understand

        pca = PCA(n_components=5)
        pca.fit(noisyDataset)
        fig, ax = plt.subplots()
        plt.plot(pca.components_.transpose())
        ax.set_title("Keeping only 5 components")
        plt.show() # This looks like crap too. I don't understand

        pca = PCA(n_components=10)
        pca.fit(noisyDataset)
        fig, ax = plt.subplots()
        plt.plot(pca.components_.transpose())
        ax.set_title("Keeping only 10 components")
        plt.show() # This looks like total crap. What is going on?

    @unittest.skipIf(skipPlots, "Skip plots")
    def testExpressOriginalBasisVectorInNewObtainedEigenvectorBase(self):
        # So I know that the components returned by PCA form a complete basis for my data set.
        # But I also know that my actual basisSet is another valid basis.  I should be able to express
        # my basisSet with the components basis set.
        # I am following https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        N = 100
        basisDimension = 5
        basisSet = self.createBasisSet(self.X, N=basisDimension, maxPeaks=3, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataset = self.createDatasetFromBasisSet(N=N, basisSet=basisSet)
        noisyDataset = self.newDatasetWithAdditiveNoise(dataset, fraction=0.1)
        
        # We keep most components to get a "perfect fit"
        componentsToKeep = 20
        pca = PCA(n_components=componentsToKeep)
        pca.fit(noisyDataset)

        # We get the coefficients for our original basis set
        basisCoefficients = pca.transform(basisSet)
        self.assertTrue(basisCoefficients.shape == (basisDimension, componentsToKeep))
        recoveredBasisSet = pca.inverse_transform(basisCoefficients)

        error = basisSet-recoveredBasisSet

        recoveredBasisSetWrong = basisCoefficients@pca.components_
        basisCoefficientsWrong = pca.transform(recoveredBasisSetWrong)

        fig, (ax1, ax2, ax3) = plt.subplots(3,figsize=(10,10))        
        ax1.plot(recoveredBasisSet.transpose())
        ax1.set_title("Recovered basis from projection")
        ax2.plot(basisSet.transpose())
        ax2.set_title("Original basis, should be the same")
        ax3.plot(error.transpose())
        ax3.set_title("Residual error")
        plt.show()

    @unittest.skipIf(skipPlots, "Skip plots")
    def testErrorAsAFunctionOfComponentsKept(self):
        # Now I will actually test the error between the computed recovered basisSet and the real basisSet
        # as a function of the number of components I keep in my eigen basis.
        # I am following https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        N = 100
        basisDimension = 5
        basisSet = self.createBasisSet(self.X, N=basisDimension, maxPeaks=3, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataset = self.createDatasetFromBasisSet(N=N, basisSet=basisSet)
        noisyDataset = self.newDatasetWithAdditiveNoise(dataset, fraction=0.1)
        
        errors = []
        componentsRange = range(101)
        for componentsToKeep in componentsRange:
            pca = PCA(n_components=componentsToKeep)
            pca.fit(noisyDataset)

            # We get the coefficients for our original basis set
            basisCoefficients = pca.transform(basisSet)
            self.assertTrue(basisCoefficients.shape == (basisDimension, componentsToKeep))
            recoveredBasisSet = pca.inverse_transform(basisCoefficients)

            error = basisSet-recoveredBasisSet
            errors.append(sum(sum(abs(error))))

        fig, ax = plt.subplots(1)
        ax.plot(componentsRange, errors,'o')
        ax.set_yscale("log")
        ax.set_title("Error with {0} original basis vectors".format(basisDimension))
        ax.set_xlabel("Number of components kept")
        plt.show()

    @unittest.skipIf(skipPlots, "Skip plots")
    def testUnderstandingPCAInverseTransform(self):
        # I thought I understood inverse_transform() but I tried to do it manually and I failed:
        # I thought I could simply do basisCoefficients@pca.components_ to recover my basis set but hat 
        # does not work. I figured it out though. See below.

        N = 100
        basisDimension = 5
        basisSet = self.createBasisSet(self.X, N=basisDimension, maxPeaks=3, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataset = self.createDatasetFromBasisSet(N=N, basisSet=basisSet)
        noisyDataset = self.newDatasetWithAdditiveNoise(dataset, fraction=0.1)
        
        # We keep most components to get a "perfect fit"
        componentsToKeep = 5
        pca = PCA(n_components=componentsToKeep)
        pca.fit(noisyDataset)

        # We get the coefficients for our original basis set
        basisCoefficients = pca.transform(basisSet)
        self.assertTrue(basisCoefficients.shape == (basisDimension, componentsToKeep))

        # I don't understand why pca.inverse_transform is not the same as basisCoefficients@pca.components_
        recoveredBasisSet = pca.inverse_transform(basisCoefficients)

        # I looked at the code for inverse_transform() and I think I figured it out:
        # https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/decomposition/_base.py#L97
        # The inverse_transform() method substracts the mean of all vectors from the vectors and it must
        # be added back when we perform the inverse transform.
        recoveredBasisSetDoneManually = basisCoefficients@pca.components_+pca.mean_

        errorFromInvTransform = recoveredBasisSet-recoveredBasisSetDoneManually
        self.assertTrue(errorFromInvTransform.all() == 0)

        fig, (ax1, ax2, ax3) = plt.subplots(3,figsize=(10,10))        
        ax1.plot(recoveredBasisSetDoneManually.transpose())
        ax1.set_title("Recovered basis from projection")
        ax2.plot(basisSet.transpose())
        ax2.set_title("Original basis, should be the same")
        ax3.plot(errorFromInvTransform.transpose())
        ax3.set_title("Residual error")
        plt.show()

    def createNoisyDataset(self, nSamples=100, basisDimension=5, maxPeaks=3, maxAmplitude=1, maxWidth=30, minWidth=5, noiseFraction=0.1 ):
        # I am sick of creating a noisy dataset every time. Here is a useful function.
        basisSet = self.createBasisSet(self.X, N=basisDimension, maxPeaks=maxPeaks, 
                                       maxAmplitude=maxAmplitude, maxWidth=maxWidth, minWidth=minWidth)
        dataset = self.createDatasetFromBasisSet(N=nSamples, basisSet=basisSet)
        noisyDataset = self.newDatasetWithAdditiveNoise(dataset, fraction=noiseFraction)
        return basisSet, noisyDataset

    def testBaseChange(self):
        # All I have left to do is to perform a base change from pca.components to my basisSet
        basisSet, dataSet = self.createNoisyDataset()
        self.assertIsNotNone(basisSet)
        self.assertIsNotNone(dataSet)
        # I will test more here later...: I want to use the pca.coefficients and transform them into 
        # the real physical basis that I know has a physical meaning.  I need to perform a base change.

if __name__ == '__main__':
    unittest.main()
