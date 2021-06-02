import unittest
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


inf = float("+inf")

A1 = 1.0
X1 = 250
W1 = 50

A2 = 1.0
X2 = 600
W2 = 50

class TestPCA(unittest.TestCase):
    X = None
    C1 = None
    C2 = None

    def testPCAIsImportingProperly(self):
        pca = PCA()
        self.assertIsNotNone(pca)

    def testSimulatedXComponent(self):
        X  = np.linspace(0,1000,1001)
        self.assertIsNotNone(X)
        self.assertTrue(len(X) != 0)

    def testSimulatedC1Component(self):
        X  = np.linspace(0,1000,1001)
        C1 = A1*np.exp(-(X-X1)**2/W1)
        self.assertIsNotNone(C1)
        self.assertEqual(len(C1), len(X))
        self.assertTrue(np.mean(C1) > 0)

    def testSimulatedC2Component(self):
        X  = np.linspace(0,1000,1001)
        C2 = A2*np.exp(-(X-X2)**2/W2)
        self.assertIsNotNone(C2)
        self.assertTrue(np.mean(C2) > 0)

    def setUp(self):
        # Once tested, we set them up every time in setUp
        self.X  = np.linspace(0,1000,1001)
        self.C1 = A1*np.exp(-(self.X-X1)**2/W1)
        self.C2 = A2*np.exp(-(self.X-X2)**2/W2)

    def testSimulatedC1Max(self):
        index = np.argmax(self.C1)
        self.assertEqual(self.X[index], X1)
        self.assertEqual(self.C1[index], A1)

    def testSimulatedC2Max(self):
        index = np.argmax(self.C2)
        self.assertEqual(self.X[index], X2)
        self.assertEqual(self.C2[index], A2)

    def createDataset(self, N):
        # Create N random combinations (0..1) of C1 and C2
        dataset = []
        for i in range(N):
            a1 = random.random()
            a2 = random.random()
            vector = a1*self.C1 + a2*self.C2
            dataset.append(vector)

        return np.stack(dataset)

    def testDatasetCreation(self):
        N = 100
        dataset = self.createDataset(N=N)
        self.assertTrue(len(dataset) == N)
        for v in dataset:
            self.assertTrue(len(v) == len(self.X))

    def newDatasetWithAdditiveNoise(self, dataset, fraction):
        noisyDataset = []
        for v in dataset:
            noisyVector = []
            for i, value in enumerate(v):
                noise = fraction * random.random() * np.max(v)
                noisyVector.append(v[i] + noise)
            noisyDataset.append(noisyVector)

        return noisyDataset

    def testNoisyDatasetCreation(self):
        N = 100
        dataset = self.createDataset(N=N)
        noisyDataset = self.newDatasetWithAdditiveNoise(dataset, 0.05)

        self.assertTrue(len(noisyDataset) == len(dataset))
        for i in range(len(dataset)):
            v = dataset[i]
            vn = noisyDataset[i]

            for j in range(len(v)):
                self.assertTrue(v[j] != vn[j])

    # I am now ready to test PCA with data
    def testPCAIsImportingProperly(self):
        pca = PCA()
        self.assertIsNotNone(pca)

    @unittest.skip("Not matplotlib")
    def testFitPCA(self):
        # I am following https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        N = 100
        dataset = self.createDataset(N=N)
        noisyDataset = self.newDatasetWithAdditiveNoise(dataset, 0.1)

        pca = PCA(n_components=2)
        pca.fit(noisyDataset)
        print(pca.singular_values_)
        print(pca.explained_variance_ratio_)
        fig, ax = plt.subplots()
        plt.plot(pca.components_.transpose())
        ax.set_title("Keeping only 2 components")
        plt.show()

        pca = PCA(n_components=5)
        pca.fit(noisyDataset)
        print(pca.singular_values_)
        print(pca.explained_variance_ratio_)
        fig, ax = plt.subplots()
        plt.plot(pca.components_.transpose())
        ax.set_title("Keeping only 5 components")
        plt.show()

        pca = PCA(n_components=10)
        pca.fit(noisyDataset)
        print(pca.singular_values_)
        print(pca.explained_variance_ratio_)
        fig, ax = plt.subplots()
        plt.plot(pca.components_.transpose())
        ax.set_title("Keeping only 10 components")
        plt.show()

    def createComponent(self, x, maxPeaks, maxAmplitude, maxWidth, minWidth):        
        N = random.randint(1, maxPeaks)
        
        intensity = np.zeros(len(x))
        for i in range(N):
            amplitude = random.uniform(0, maxAmplitude)
            width = random.uniform(minWidth, maxWidth)
            center = random.choice(x)
            intensity += amplitude*np.exp(-(x-center)**2/width**2)

        return intensity

    def testCreateSpectrum(self):

        component = self.createComponent(self.X, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5)
        self.assertIsNotNone(component)
        self.assertTrue(len(component) == len(self.X))
        # plt.plot(self.X, component)
        # plt.show()


    def createBasisSet(self, x, N, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5):
        basisSet = []
        for i in range(N):
            component = self.createComponent(x, maxPeaks, maxAmplitude, maxWidth, minWidth)
            self.assertIsNotNone(component)
            self.assertTrue(len(component) == len(x))
            basisSet.append(component)

        return np.array(basisSet)

    def testCreateBaseComponents(self, ):
        basisSet = self.createBasisSet(self.X, N=5, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5)
        self.assertTrue(basisSet.shape == (5, len(self.X)))

    def createDatasetFromBasisSet(self, N, basisSet):
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
        N = 100
        basisSet = self.createBasisSet(self.X, N=5, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataset = self.createDatasetFromBasisSet(N=N, basisSet=basisSet)
        dataset = self.newDatasetWithAdditiveNoise(dataset, fraction=0.1)

        self.assertTrue(len(dataset) == N)
        for v in dataset:
            self.assertTrue(len(v) == len(self.X))
    @unittest.skip("Not mplots")
    def testFitPCAWithMoreComplexBasisSet(self):
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
        plt.show()

        pca = PCA(n_components=5)
        pca.fit(noisyDataset)
        fig, ax = plt.subplots()
        plt.plot(pca.components_.transpose())
        ax.set_title("Keeping only 5 components")
        plt.show()

        pca = PCA(n_components=10)
        pca.fit(noisyDataset)
        fig, ax = plt.subplots()
        plt.plot(pca.components_.transpose())
        ax.set_title("Keeping only 10 components")
        plt.show()

    def testExpressOriginalBasisVectorInNewObtainedEigenvectorBase(self):
        # I am following https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        N = 100
        basisDimension = 5
        basisSet = self.createBasisSet(self.X, N=basisDimension, maxPeaks=3, maxAmplitude=1, maxWidth=30, minWidth=5)
        dataset = self.createDatasetFromBasisSet(N=N, basisSet=basisSet)
        noisyDataset = self.newDatasetWithAdditiveNoise(dataset, fraction=0.1)
        
        # We keep all components to get a "perfect fit"
        componentsToKeep = 6
        pca = PCA(n_components=componentsToKeep)
        pca.fit(noisyDataset)

        # We get the coefficients for our original basis set
        basisCoefficients = pca.transform(basisSet)
        self.assertTrue(basisCoefficients.shape == (basisDimension, componentsToKeep))
        recoveredBasisSet = pca.inverse_transform(basisCoefficients)
        recoveredBasisSetWrong = basisCoefficients@pca.components_
        basisCoefficientsWrong = pca.transform(recoveredBasisSetWrong)
        print(basisCoefficientsWrong/basisCoefficients)
        fig, (ax1, ax2, ax3) = plt.subplots(3)        
        ax1.plot(recoveredBasisSet.transpose())
        ax1.set_title("Recovered basis from projection")
        ax2.plot(basisSet.transpose())
        ax2.set_title("Original basis, should be the same")
        ax3.plot(pca.components_.transpose())
        ax3.set_title("Principal components")
        plt.show()


if __name__ == '__main__':
    unittest.main()
