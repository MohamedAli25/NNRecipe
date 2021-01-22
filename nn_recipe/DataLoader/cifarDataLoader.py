from dataLoader import *
import numpy as np


class CifarDataLoader(DataLoader):
    _resources = [
        ('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
         'c58f30108f718f92721af3b95e74349a')
        # ('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 'eb9058c3a382ffc7106e4002c42a8d85'),
    ]

    def __init__(self, root_path="../../cifar-10-batches-py", download=False):
        """ Loads the dataset and downloads it if required
        Args:
            root (str): The path to the dataset directory
            download (bool): A flag for downloading the dataset
                            (default is False)                      
        """
        # os.chdir(root_path)
        # for (url, md5) in self._resources:
        #     filename = url.rpartition('/')[2]
        #     if filename not in os.listdir(root_path) and download is True:
        #         urllib.request.urlretrieve(''.join((url, tar)), os.path.join(root_path, tar))
        #
        #     with tarfile.open(os.path.join(root_path, filename)) as tar_object:
        #         members = [file for file in tar_object if file.name in files]
        self.rootPath = root_path
        self.trainData = None
        self.trainLabels = None
        self.testData: np.ndarray = None
        self.testLabels = None
        self.labelNames = None
        # return members

    def unpickle(self, file: str):
        with open("{}/{}".format(self.rootPath, file), 'rb') as fo:
            dataDict = pickle.load(fo, encoding='bytes')
        return dataDict

    def load(self):
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

        # Load Test Data
        testDataFilename = "test_batch"
        testDataDict = self.unpickle(testDataFilename)
        self.testData = testDataDict[b"data"]
        self.testLabels = testDataDict[b"labels"]
        self.testData = self.testData.reshape((len(self.testData), 3, 32, 32))
        self.testData = self.testData.transpose(0, 2, 3, 1).astype(np.float32)
        self.testLabels = np.array(self.testLabels)

        # Assertions
        assert len(self.labelNames) == 10
        assert self.trainData.shape == (50000, 32, 32, 3)
        assert self.trainLabels.shape == (50000,)
        assert self.testData.shape == (10000, 32, 32, 3)
        assert self.testLabels.shape == (10000,)


    def getData(self):
        pass

    def get_label_names(self):
        return self.labelNames

    def get_train_data(self):
        return self.trainData

    def get_train_labels(self):
        return self.trainLabels

    def get_test_data(self):
        return self.testData

    def get_test_labels(self):
        return self.testLabels


if __name__ == "__main__":
    # cfDL = CifarDataLoader('E:\Engineering_courses\Senior\NN\Project', download=True)
    # print(cfDL, type(cfDL))
    cifarDataLoader = CifarDataLoader()
    cifarDataLoader.load()
    print("Labels: " + str(cifarDataLoader.labelNames))
    print(cifarDataLoader.trainData.shape)
    print(cifarDataLoader.trainLabels.shape)
    print(cifarDataLoader.testData.shape)
    print(cifarDataLoader.testLabels.shape)
