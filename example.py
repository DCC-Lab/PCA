import unittest
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
import csv

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

# Run tests in order they are written
unittest.TestLoader.sortTestMethodsUsing = None

class TestPCA(unittest.TestCase):

    def test01PCAIsImportingProperly(self):
        # Is my module installed properly at least?
        pca = PCA()
        self.assertIsNotNone(pca)

    def testRead(self):
        with open('ex1.csv', newline='') as f:
            reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
            for row in reader:
                pass#print(row)

    def dataset(self, filepath):
        dataset = np.array()
        with open(filepath, newline='') as f:
            reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
            for row in reader:
                data = []
                for x in row:
                    data.append(float(x))
                dataset.append(data)    
        return np.array(dataset)

    # def testEx1(self):
    #     pca = PCA(n_components=3, whiten=False)
    #     dataset = self.dataset('ex1.csv')
    #     pca.fit(dataset)
    #     # print(np.array(dataset).shape)
    #     # print(pca.explained_variance_ratio_)
    #     # print(pca.components_)

    #     pcaW = PCA(n_components=3, whiten=True)
    #     dataset = self.dataset('ex1.csv')
    #     pcaW.fit(dataset)

    #     plt.plot(pca.components_.T)
    #     plt.plot(pcaW.components_.T)
    #     plt.show()

    def testPCA(self):
        pca = PCA(n_components=3, whiten=True)
        dataset = self.dataset('pca.csv')
        pca.fit(dataset)
        # print(np.array(dataset).shape)
        # print(pca.explained_variance_ratio_)
        # print(pca.components_)
        plt.plot(pca.components_.T)
        plt.show()

    def test2D(self):
        pca = PCA(n_components=3, whiten=True)
        dataset = self.dataset('pca.csv')
        normData = (dataset-np.mean(dataset,axis=0))/np.std(dataset,axis=0)
        pca.fit(normData)


        for student in normData:
            plt.plot(student[0],student[3],'ko')

        plt.show()

    def test3D(self):
        pca = PCA(n_components=3, whiten=True)
        dataset = self.dataset('pca.csv')
        pca.fit(dataset)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for student in dataset:
            ax.scatter(student[0],student[1],student[2], marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    def testPCANorm(self):
        pca = PCA(n_components=3)
        dataset = self.dataset('ex1.csv')
        print(dataset,np.mean(dataset,axis=1))
        normData = (dataset-np.mean(dataset,axis=0))/np.std(dataset,axis=0)
        pca.fit(dataset)
        # print(np.array(dataset).shape)
        # print(pca.explained_variance_ratio_)
        # print(pca.components_)
        plt.plot(pca.components_.T)
        plt.show()


if __name__ == '__main__':
    unittest.main()
