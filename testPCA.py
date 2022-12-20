import unittest
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LabPCA(PCA):
    def transform_noncentered(self, X):
        originCoefficients = np.zeros(shape=X.shape)
        return self.transform(X)-self.transform(originCoefficients)

    @property
    def components_noncentered_(self):
        return self.components_ + self.mean_


# I will use these later, it is easier if I put them here:
A1 = 1.0 # amplitude of peak1
X1 = 250 # center
W1 = 50  # gaussian width

A2 = 1.0 # same for peak 2
X2 = 600
W2 = 50

skipPlots = True # It is annoying to have to press 'q' to dismiss the plots
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
noise and see how PCA handles them: I want to recover the concentrations of these
analytes.
"""

# Run tests in order they are written
unittest.TestLoader.sortTestMethodsUsing = None

class TestPCA(unittest.TestCase):
    X = None
    C1 = None
    C2 = None

    def test01PCAIsImportingProperly(self):
        # Is my module installed properly at least?
        pca = PCA()
        self.assertIsNotNone(pca)

    def test02SimulatedXComponent(self):
        # I will need spectra, so let me get started to make sure I can at least do that.
        # Start with a "wavelength" range.
        X  = np.linspace(0,1000,1001)
        self.assertIsNotNone(X)
        self.assertTrue(len(X) != 0)
        # I want to have 0,1,2....
        self.assertEqual(X[0], 0)
        self.assertEqual(X[1], 1)

    def test03SimulatedC1Component(self):
        # Create a fake C1 spectrum, my "first analyte"
        X  = np.linspace(0,1000,1001)
        C1 = A1*np.exp(-(X-X1)**2/W1)
        self.assertIsNotNone(C1)
        self.assertEqual(len(C1), len(X))
        self.assertTrue(np.mean(C1) > 0)

    def test04SimulatedC2Component(self):
        # Create another C2 spectrum, my second analyte
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

    def test05SimulatedC1Max(self):
        # Test that the values are fine: centered on X1, amplitude A1
        index = np.argmax(self.C1)
        self.assertEqual(self.X[index], X1)
        self.assertEqual(self.C1[index], A1)

    def test06SimulatedC2Max(self):
        # Test that the values are fine: centered on X2, amplitude A2
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
            spectrum = a1*self.C1 + a2*self.C2
            dataset.append(spectrum)

        return np.stack(dataset)

    def test07DatasetCreation(self):
        # Test my creation function above.
        N = 100
        dataset = self.createDataset(N=N)
        self.assertTrue(len(dataset) == N)
        for v in dataset:
            self.assertTrue(len(v) == len(self.X))

    def newDatasetWithAdditiveNoise(self, dataset, fraction):
        # Ok, I have several spectra in a dataset, I will add noise to make it real
        noisyDataset = []
        for v in dataset:
            noisyVector = []
            for i, value in enumerate(v):
                noise = fraction * random.random() * np.max(v)
                noisyVector.append(v[i] + noise)
            noisyDataset.append(noisyVector)

        return noisyDataset

    def test8NoisyDatasetCreation(self):
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

    def test09PCAIsImportingProperly(self):
        # I am now ready to test PCA with data. Is it still well-installed?
        # (ah ah)
        pca = PCA()
        self.assertIsNotNone(pca)

    @unittest.skipIf(skipPlots, "Skip plots")
    def test10FitPCA(self):
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

    def test11CreateSpectrum(self):
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

    def test12CreateBaseComponents(self):
        # Is the function I just wrote working?
        basisSet = self.createBasisSet(self.X, N=5, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5)
        self.assertTrue(basisSet.shape == (5, len(self.X)))


    def createDatasetFromBasisSet(self, N, basisSet):
        # Alright, I have a basis set but now I want a data set

        # I am a bit confused with the numpy dimensions (i.e. the shape) and indices.  Which way do they go?
        # I think it's this:
        # shape = (# base, #spectral_pts)
        
        m, nPts = basisSet.shape
        C = np.random.rand(m, N)

        return (basisSet.T@C).T, C

    def test13DatasetCreationFromBasisSet(self):
        # Is this createDatasetFromBasisSet working as expected?
        N = 100
        basisSet = self.createBasisSet(self.X, N=5, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataset, concentration = self.createDatasetFromBasisSet(N=N, basisSet=basisSet)
        dataset = self.newDatasetWithAdditiveNoise(dataset, fraction=0.1)

        self.assertTrue(len(dataset) == N)
        for v in dataset:
            self.assertTrue(len(v) == len(self.X))

    def test14MeanSpectrumCalculation(self):
        # I should be able to recover the mean from a simple calculation
        N = 100
        basisSet = self.createBasisSet(self.X, N=5, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataset, concentration = self.createDatasetFromBasisSet(N=N, basisSet=basisSet)
        # dataset = self.newDatasetWithAdditiveNoise(dataset, fraction=0.01)

        meanConcentration = np.mean(concentration, axis=1)
        meanSpectrum = basisSet.T@meanConcentration
        pca = PCA(n_components=6)
        pca.fit(dataset)
        self.assertEqual( (pca.mean_.T-meanSpectrum.T).all(), 0)

    @unittest.skipIf(skipPlots, "Skip plots")
    def test15FitPCAWithMoreComplexBasisSet(self):
        # Alright, now we are really entering the real stuff:
        # I should be able to use PCA and get "eigenvectors" that should resemble my basis set
        # I am following https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        N = 1000
        basisSet = self.createBasisSet(self.X, N=5, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataset, concentration = self.createDatasetFromBasisSet(N=N, basisSet=basisSet)
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
    def test16ExpressOriginalBasisVectorInNewObtainedEigenvectorBase(self):
        # So I know that the components returned by PCA form a complete basis for my data set.
        # But I also know that my actual basisSet is another valid basis.  I should be able to express
        # my basisSet with the components basis set.
        # I am following https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        N = 100
        basisDimension = 5
        basisSet = self.createBasisSet(self.X, N=basisDimension, maxPeaks=3, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataset, concentration = self.createDatasetFromBasisSet(N=N, basisSet=basisSet)
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
    def test17ErrorAsAFunctionOfComponentsKept(self):
        # Now I will actually test the error between the computed recovered basisSet and the real basisSet
        # as a function of the number of components I keep in my eigen basis.
        # I am following https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        N = 100
        basisDimension = 5
        basisSet = self.createBasisSet(self.X, N=basisDimension, maxPeaks=3, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataset, concentration = self.createDatasetFromBasisSet(N=N, basisSet=basisSet)
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
    def test18UnderstandingPCAInverseTransform(self):
        # I thought I understood inverse_transform() but I tried to do it manually and I failed:
        # I thought I could simply do basisCoefficients@pca.components_ to recover my basis set but hat 
        # does not work. I figured it out though. See below.

        N = 100
        basisDimension = 5
        basisSet = self.createBasisSet(self.X, N=basisDimension, maxPeaks=3, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataset, concentration = self.createDatasetFromBasisSet(N=N, basisSet=basisSet)
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

    def createNormalizedBasisSet(self, x, N, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5):
        # I think it would be useful to have a normalized basis set.  Let me see what I get.
        # I figured out later I could have used np.linalg.norm()
        basisSet = []
        for i in range(N):
            component = self.createComponent(x, maxPeaks, maxAmplitude, maxWidth, minWidth)            
            self.assertIsNotNone(component)
            self.assertTrue(len(component) == len(x))
            magnitudeSquared = np.dot(component, component)
            self.assertTrue(magnitudeSquared>0)
            component /= np.sqrt(magnitudeSquared)
            basisSet.append(component)

        return np.array(basisSet)

    def test19NormalizationOfBasisSet(self):
        # I calculate the norm as the sqrt() of the dot product with itself.
        component = self.createComponent(self.X, maxPeaks=3, maxAmplitude=1, maxWidth=30, minWidth=10)

        magnitudeSquared = np.dot(component.T, component)
        component /= np.sqrt(magnitudeSquared)
        magnitudeSquared = np.sqrt(np.dot(component.T, component))
        self.assertAlmostEqual(magnitudeSquared, 1.0)

    def test20NonOrthogonalityOfBasisSet(self):
        # Components are certainly not orthogonal: if they have peaks that overlap, they are not.
        # If they have completely separate peaks, they are.
        component1 = self.createComponent(self.X, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=10)
        magnitudeSquared = np.dot(component1.T, component1)
        component1 /= np.sqrt(magnitudeSquared)

        component2 = self.createComponent(self.X, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=10)
        magnitudeSquared = np.dot(component2.T, component2)
        component2 /= np.sqrt(magnitudeSquared)

        magnitudeSquared = np.dot(component1.T, component2)
        self.assertTrue(magnitudeSquared != 0)

    def createNormalizedNoisyDataset(self, nSamples=100, basisDimension=5, maxPeaks=3, maxAmplitude=1, maxWidth=30, minWidth=5, noiseFraction=0.1 ):
        # I am sick of creating a noisy dataset every time. Here is a useful function.
        basisSet = self.createNormalizedBasisSet(self.X, N=basisDimension, maxPeaks=maxPeaks, 
                                       maxAmplitude=maxAmplitude, maxWidth=maxWidth, minWidth=minWidth)
        dataset, concentration = self.createDatasetFromBasisSet(N=nSamples, basisSet=basisSet)
        noisyDataset = self.newDatasetWithAdditiveNoise(dataset, fraction=noiseFraction)
        return np.array(basisSet), np.array(noisyDataset), np.array(concentration)

    def test21AgainWithNormalizedBasisSetPCAreNormalized(self):
        basisSet, noisyDataset, concentration = self.createNormalizedNoisyDataset()

        componentsToKeep = 5
        pca = PCA(n_components=componentsToKeep)
        pca.fit(noisyDataset)
        for pc in pca.components_:
            magnitude = np.sqrt(np.dot(pc.T,pc))
            self.assertAlmostEqual(magnitude, 1.0)

    def testValidateProjectionsBasisSetOntoPrincipalComponents(self):
        basisSet, noisyDataset, concentration = self.createNormalizedNoisyDataset()

        componentsToKeep = 5
        pca = PCA(n_components=componentsToKeep)
        pca.fit(noisyDataset)

        for basis in basisSet:
            magnitude = np.sqrt(np.dot(basis,basis))
            self.assertTrue(magnitude > 0)
            for pc in pca.components_:
                scalarProduct = np.dot(pc,basis)
                self.assertTrue(scalarProduct != 0)

    @unittest.skipIf(skipPlots, "Skip plots")
    def test22BaseChange(self):
        # All I have left to do is to perform a base change from pca.components to my basisSet
        # I am following Greenberg 10.7 but my basis set is not Orthonormal and is also translated. 
        # Let's be careful.
        basisSet, dataSet, concentration = self.createNormalizedNoisyDataset()
        self.assertIsNotNone(basisSet)
        self.assertIsNotNone(dataSet)
        componentsToKeep = 5
        pca = PCA(n_components=componentsToKeep)
        pca.fit(dataSet)

        baseChangeMatrix = np.ndarray(shape=(componentsToKeep,5))
        for i, toBaseVector in enumerate(basisSet):
            for j, fromBaseVector in enumerate(pca.components_):
                qij = np.dot(toBaseVector,fromBaseVector)
                self.assertTrue(qij != 0)
                baseChangeMatrix[i,j] = qij

        self.assertTrue(baseChangeMatrix.shape == (5,5))
        PC1Coefficients = np.zeros(shape=(5,1))
        PC1Coefficients[0] = 1

        physicalComponents = baseChangeMatrix@PC1Coefficients
        pc1InOriginalBasis = basisSet.T@physicalComponents
        pc1InOriginalBasis /= np.dot(pc1InOriginalBasis.T,pc1InOriginalBasis)

        plt.plot(pca.components_[0],label='Principal Component 1')
        plt.plot(basisSet[0]*physicalComponents[0],label='{0} x base1'.format(physicalComponents[0]))
        plt.plot(basisSet[1]*physicalComponents[1],label='{0} x base2'.format(physicalComponents[1]))
        plt.plot(basisSet[2]*physicalComponents[2],label='{0} x base3'.format(physicalComponents[2]))
        plt.plot(basisSet[3]*physicalComponents[3],label='{0} x base4'.format(physicalComponents[3]))
        plt.plot(basisSet[4]*physicalComponents[4],label='{0} x base5'.format(physicalComponents[4]))

        plt.legend()
        plt.show()

    @unittest.expectedFailure
    def test23NonOrthogonalBaseChangeTesting(self):
        # My basis set is not orthogonal, therefore this test will fail
        # The equation for the base change matrix qij is not just the dot product of two vectors

        basisSet, dataSet, concentration = self.createNormalizedNoisyDataset()
        self.assertIsNotNone(basisSet)

        for i, toBaseVector in enumerate(basisSet):
            for j, fromBaseVector in enumerate(basisSet):
                qij = np.dot(toBaseVector.T,fromBaseVector)
                if i == j:
                    self.assertAlmostEqual(qij, 1.0, 3)
                else:
                    self.assertAlmostEqual(qij, 0.0, 3)


    @unittest.skip("sometimes will fail of course")
    def test24ProbablyOrthogonalBaseChangeTesting(self):
        # My basis set is not orthogonal with large peaks, therefore
        # Let me make an orthogonal basis with very narrow peaks.
        # The probabiblity that two peaks overlap will be small and
        # my basis should be orthogonal, but the test may fail sometimes.
        
        basisSet, dataSet, concentration = self.createNormalizedNoisyDataset(maxPeaks=1, maxWidth=2, minWidth=1)
        self.assertIsNotNone(basisSet)

        for i, toBaseVector in enumerate(basisSet):
            for j, fromBaseVector in enumerate(basisSet):
                qij = np.dot(toBaseVector.T,fromBaseVector)
                if i == j:
                    self.assertAlmostEqual(qij, 1.0, 3)
                else:
                    self.assertAlmostEqual(qij, 0.0, 3)

    def test25NonOrthogonalBaseChangeValidatingRealCorrectEquation(self):
        # So equation (10) in greenberg is not valid with non orthogonal basis
        # The thing is, I know how to express my original vectors on the principal
        # components basis, it is simply pca.transform(basisSet)
        
        basisSet, dataSet, concentration = self.createNormalizedNoisyDataset()
        self.assertIsNotNone(basisSet)
        componentsToKeep = 5
        pca = PCA(n_components=componentsToKeep)
        pca.fit(dataSet)

        Q = pca.transform(basisSet)
        invQ = np.linalg.inv(Q)

    @unittest.skip("No plots")
    def test26ShowTranslatedPCs(self):
        basisSet, dataSet, concentration = self.createNormalizedNoisyDataset()
        componentsToKeep = 5
        pca = PCA(n_components=componentsToKeep)
        pca.fit(dataSet)

        pcs = pca.components_+pca.mean_
        plt.plot(pcs.T,label='Translated PCs')
        plt.plot(pca.mean_,label='Mean')
        plt.legend()
        plt.show()

    def test27MatrixConstruction(self):
        # I need to build various matrices for testing
        one = np.ones(shape=(5,5))
        for i in range(5):
            for j in range(5):
                self.assertEqual(one[i,j], 1.0)  

        half = np.ones(shape=(5,5))
        half *= 0.5
        for i in range(5):
            for j in range(5):
                self.assertEqual(half[i,j], 0.5)

        identity = np.identity(5)
        self.assertIsNotNone(identity-half)        

    @unittest.skip
    def test28BasisV0SpectrumInPCSpace(self):
        # This attempt failed.  I was wrong. Keeping it for historical reasons.
        basisSet, dataSet, concentration = self.createNormalizedNoisyDataset(noiseFraction=0.0)
        self.assertIsNotNone(basisSet)
        self.assertIsNotNone(dataSet)
        componentsToKeep = 5
        pca = PCA(n_components=componentsToKeep)
        self.assertTrue(basisSet.shape == (5, 1001))
        self.assertTrue(dataSet.shape == (100, 1001))
        pca.fit(dataSet)

        v0 = basisSet[0,:].squeeze()
        self.assertTrue(v0.shape == (1001,))

        v0Coefficients = pca.transform(v0.reshape(1,-1)).squeeze()
        self.assertTrue(v0Coefficients.shape == (5,))

        v0PCA  = (pca.components_.T@v0Coefficients + pca.mean_).squeeze()
        self.assertTrue(v0Coefficients.shape == (componentsToKeep,))
        v0Orig = (basisSet.T@(1,0,0,0,0)).squeeze()

        plt.plot(v0,label="v0")
        plt.plot(v0PCA,label="v0pca")
        # plt.plot(dataSet[0,:],label="V0dataset")
        plt.plot(v0Orig,label="v0Orig")
        plt.legend()
        plt.show()
        self.assertTrue( (v0-v0PCA).all() == 0)
        self.assertTrue( (v0-v0Orig).all() == 0)

        # identity = np.identity(5)
        # baseChangeMatrixFromPCToOriginal = (identity-meanConcentration)@np.linalg.inv(baseChangeMatrixFromOriginalToPC)

        # sample1 = np.array(dataSet[0])
        # concentrationSample1 = np.array(concentration[:,0])

        # coefficientsInPC = pca.transform(sample1.reshape(1,-1))
        # physicalComponents = baseChangeMatrixFromPCToOriginal.T@coefficientsInPC.T
        # print(physicalComponents, concentrationSample1)
        
        # sample1InOriginalBasis = basisSet.T@concentrationSample1
        # sample1InOriginalBasis = sample1InOriginalBasis.squeeze()

    @unittest.skip
    def test29BasisSpectrumInOriginalSpace(self):
        # This attempt also failed.  I was wrong again. Keeping it for historical reasons.
        basisSet, dataSet, concentration = self.createNormalizedNoisyDataset(noiseFraction=0.0)
        self.assertIsNotNone(basisSet)
        self.assertIsNotNone(dataSet)
        componentsToKeep = 5
        pca = PCA(n_components=componentsToKeep)
        self.assertTrue(basisSet.shape == (5, 1001))
        self.assertTrue(dataSet.shape == (100, 1001))
        pca.fit(dataSet)

        meanBasis = np.stack((pca.mean_,pca.mean_,pca.mean_,pca.mean_,pca.mean_))

        c = pca.transform(dataSet)
        print(c.shape)
        a = pca.transform(basisSet-meanBasis)
        print(a.shape)

        aInv = np.linalg.inv(a)
        print(aInv.shape)
        deltaC = aInv@c.T
        print(deltaC.shape)
        print(deltaC[:,0].squeeze() + np.mean(concentration,axis=1))
        print(concentration[:,0])
        self.assertTrue(aInv.shape == (5,5))

        # coefficientsInPC = pca.transform(sample1.reshape(1,-1))
        # physicalComponents = baseChangeMatrixFromPCToOriginal.T@coefficientsInPC.T
        # print(physicalComponents, concentrationSample1)
        
        # sample1InOriginalBasis = basisSet.T@concentrationSample1
        # sample1InOriginalBasis = sample1InOriginalBasis.squeeze()

    @unittest.skip
    def test30EasyBaseChange(self):
        # Do I understand what the hell I am trying to do?
        # Can I even do a simple base change with non-orthogonal base?
        q = np.array( [[1,1,1],[0,1,1],[0,0,1]])
        invQ = np.linalg.inv(q)
        print(q)
        print(invQ) # looks good, hard to test, ok I am not stupid.

    @unittest.skip("This is a failed attempt at getting something to work")
    def test31RecoverConcentrationByProjectingInPCSpaceOnOriginalBasis(self):
        # Yet again, this failed. Will I ever succeed? Yes, see below.
        basisSet, dataSet, concentration = self.createNormalizedNoisyDataset(noiseFraction=0.0)
        self.assertIsNotNone(basisSet)
        self.assertIsNotNone(dataSet)
        self.assertTrue(dataSet.shape == (100, 1001))
        self.assertTrue(basisSet.shape == (5, 1001))
        self.assertTrue(concentration.shape == (5, 100))
        meanConcentration = np.mean(concentration, axis=1)
        self.assertTrue(meanConcentration.shape == (5,))

        componentsToKeep = 6
        pca = PCA(n_components=componentsToKeep)
        pca.fit(dataSet)

        origin = np.zeros(1001)
        originCoeffs = pca.transform(origin.reshape(1,-1))

        pcaDataCoefficients  = pca.transform(dataSet)
        pcaBasisCoefficients = pca.transform(basisSet)
        self.assertTrue(pcaDataCoefficients.shape == (100, componentsToKeep))
        self.assertTrue(pcaBasisCoefficients.shape == (5, componentsToKeep))

        sample0 = pcaDataCoefficients[0,:].squeeze()
        print("\nNorm sample0 ", np.linalg.norm(sample0))
        base0   = pcaBasisCoefficients[0,:].squeeze()
        print("Norm base0 ", np.linalg.norm(base0))
        delta = np.dot(sample0, base0)

        print("Dot product ", delta)
        print("M coeffs ", originCoeffs[0,0])
        print("Concentration sample0 ", concentration[0,0])
        print("Mean concentration ", meanConcentration[0])

    def test32WhatIsMeanInPCACoordinates(self):
        # The sklearn module always substracts the mean of all spectra, and this causes issues
        # when trying to make a base change.  I am looking for the mean_ vector expressed 
        # in pca coordinates so I can add it to the components I get from transform()
        basisSet, dataSet, concentration = self.createNormalizedNoisyDataset(noiseFraction=0.0)
        self.assertIsNotNone(basisSet)
        self.assertIsNotNone(dataSet)
        self.assertTrue(dataSet.shape == (100, 1001))
        self.assertTrue(basisSet.shape == (5, 1001))
        self.assertTrue(concentration.shape == (5, 100))

        componentsToKeep = 6
        pca = PCA(n_components=componentsToKeep)
        pca.fit(dataSet)
        origin = np.zeros(1001)
        originCoeffs = pca.transform(origin.reshape(1,-1))
        # print(originCoeffs)

    def test33RecoverConcentrationWithBaseChange(self):
        """
        Finally, a successful strategy: I can recover the concentrations
        by projecting my coefficients for a spectrum onto my original
        spectra.
        """
        basisSet, dataSet, concentration = self.createNormalizedNoisyDataset(noiseFraction=0.0)
        # self.assertIsNotNone(basisSet)
        # self.assertIsNotNone(dataSet)
        # meanConcentration = np.mean(concentration,axis=1)
        # meanConcentrationMatrix = np.array([meanConcentration]*100)
        # meanDataSet = basisSet.T@meanConcentrationMatrix.T
        # self.assertAlmostEqual( meanDataSet.all(), np.mean(dataSet).all(), 3)
        # centeredDataSet = dataSet - np.mean(dataSet)
        

        componentsToKeep = 5
        pca = PCA(n_components=componentsToKeep)
        self.assertTrue(dataSet.shape == (100, 1001))
        self.assertTrue(basisSet.shape == (5, 1001))
        self.assertTrue(concentration.shape == (5, 100))
        pca.fit(dataSet)

        originBasis = np.zeros(shape=(5, 1001))
        originDataSet = np.zeros(shape=(100, 1001))

        pcaDataCoefficients = pca.transform(dataSet)-pca.transform(originDataSet)
        pcaBasis = pca.transform(basisSet)-pca.transform(originBasis)
        pcaBasisInv = np.linalg.inv(pcaBasis) # must have 5 components and 5 base vectors

        recoveredDataSet = pca.components_.T@pcaDataCoefficients.T
        sample0Coeff = pcaDataCoefficients[0,:]
        self.assertTrue(sample0Coeff.shape==(5,))

        recoveredConcentrations = pcaBasisInv.T@sample0Coeff
        print("\nRecovered concentrations from PCA base change: ",recoveredConcentrations)
        print("Original concentrations: ",concentration[:,0])
        self.assertAlmostEqual( recoveredConcentrations.all(), concentration[:,0].all(), 3)


    def test40WithSubclassExampleGraphsAndData(self):
        # Final example with the subclass for cleaner code

        basisSet_bj = self.createBasisSet(self.X, N=5, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataSet_ij, concentration_ik = self.createDatasetFromBasisSet(100, basisSet_bj)        
        # dataSet_ij is now a simulated dataset of 100 spectra coming from 5 analytes mixed in various concentrations
        # basisSet_bj is their individual spectra

        pca = LabPCA(n_components=5)
        pca.fit(dataSet_ij)

        # Look at non-centered components
        plt.plot(pca.components_noncentered_.T)
        plt.set_title("Principal components (non-centered)")
        plt.show()

        # To avoid confusion, indices (i,b,j,k,p) represent:
        # i = sample #
        # b = basis #
        # j = feature #
        # k = concentration #
        # p = principal coefficient #
        b_bp = pca.transform_noncentered(basisSet_bj)
        s_ip = pca.transform_noncentered(dataSet_ij)
        s_pi = s_ip.T
        invb_pb = np.linalg.pinv(b_bp)
        invb_bp = invb_pb.T

        recoveredConcentrations_ki = (invb_bp@s_pi).T
        expectedConcentrations_ki = concentration_ik.T
        print("Expected concentrations (first four only):\n", expectedConcentrations_ki[0:3])
        print("Recovered concentrations (first four only):\n", recoveredConcentrations_ki[0:3])

        everythingBelowThreshold = ((expectedConcentrations_ki-recoveredConcentrations_ki) ).all() < 1e-7
        self.assertTrue(everythingBelowThreshold )
        print("Minimal differences: ", everythingBelowThreshold)

if __name__ == '__main__':
    unittest.main()
