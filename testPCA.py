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

A2 = 0.5
X2 = 600
W2 = 40

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

    def testFitPCA(self):

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


if __name__ == '__main__':
    unittest.main()
