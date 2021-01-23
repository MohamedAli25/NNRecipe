import os
import pickle
import gzip
import shutil
import tarfile

from dataLoader import *
import numpy as np


class CifarDataLoader(DataLoader):
    RESOURCES = [
        ('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
         'c58f30108f718f92721af3b95e74349a')
        # ('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 'eb9058c3a382ffc7106e4002c42a8d85'),
    ]

    def __init__(self, rootPath="../../cifar-10-batches-py", valRatio=0.177, download=False):
        """ Loads the dataset and downloads it if required
        Args:
            root (str): The path to the dataset directory
            download (bool): A flag for downloading the dataset
                            (default is False)                      
        """
        super(CifarDataLoader, self).__init__(rootPath, CifarDataLoader.RESOURCES, valRatio, download)
        self.labelNames = None

    def unpickle(self, file: str):
        with open("{}/{}".format(self.rootPath, file), 'rb') as fo:
            dataDict = pickle.load(fo, encoding='bytes')
        return dataDict

    def load(self):
        # Unzip the tar file
        tarfilePath = "{}/{}".format(self.rootPath, "cifar-10-python.tar.gz")
        tar = tarfile.open(tarfilePath)
        tarfileContentDest = os.path.join(self.rootPath, "..")
        tar.extractall(tarfileContentDest)
        tar.close()
        # Load Label Names
        batchesMeta = self.unpickle("batches.meta")
        self.labelNames = np.array(batchesMeta[b"label_names"])

        # Load Train Data
        trainDataFilenames = ["data_batch_{}".format(i) for i in range(1, 6)]
        trainDataDicts = [self.unpickle(trainDataFilename) for trainDataFilename in trainDataFilenames]
        for i in range(5):
            if i == 0:
                self.trainData = trainDataDicts[i][b"data"]
                self.trainLabels = trainDataDicts[i][b"labels"]
            else:
                self.trainData = np.vstack((self.trainData, trainDataDicts[i][b"data"]))
                self.trainLabels = self.trainLabels + trainDataDicts[i][b"labels"]
        self.trainData = self.trainData.reshape((len(self.trainData), 3, 32, 32))
        self.trainData = self.trainData.transpose(0, 2, 3, 1).astype(np.float32)
        self.trainLabels = np.array(self.trainLabels)

        # Split Train Data To Train & Validate
        numOfTrainImage = int(self.trainData.shape[0] * (1 - self.valRatio))
        self.trainData, self.validationData = self.trainData[:numOfTrainImage, :, :, :], \
                                              self.trainData[numOfTrainImage:, :, :, :]
        self.trainLabels, self.validationLabels = self.trainLabels[:numOfTrainImage], \
                                              self.trainLabels[numOfTrainImage:]

        # Load Test Data
        testDataFilename = "test_batch"
        testDataDict = self.unpickle(testDataFilename)
        self.testData = testDataDict[b"data"]
        self.testLabels = testDataDict[b"labels"]
        self.testData = self.testData.reshape((len(self.testData), 3, 32, 32))
        self.testData = self.testData.transpose(0, 2, 3, 1).astype(np.float32)
        # (50000, 32, 32, 3)
        self.testLabels = np.array(self.testLabels)


if __name__ == "__main__":
    # cfDL = CifarDataLoader('E:\Engineering_courses\Senior\NN\Project', download=True)
    # print(cfDL, type(cfDL))
    cifarDataLoader = CifarDataLoader(download=False)
    cifarDataLoader.load()
    print("Labels: " + str(cifarDataLoader.get_label_names()))
    print(cifarDataLoader.get_train_data().shape)
    print(cifarDataLoader.get_train_labels().shape)
    print(cifarDataLoader.get_validation_data().shape)
    print(cifarDataLoader.get_validation_labels().shape)
    print(cifarDataLoader.get_test_data().shape)
    print(cifarDataLoader.get_test_labels().shape)
