import os
import requests


class DataLoader:
    def __init__(self, rootPath, resources, valRatio=0.177, download=False):
        self.rootPath = rootPath
        self.valRatio = valRatio
        self.resources = resources
        self.trainData = None
        self.trainLabels = None
        self.validationData = None
        self.validationLabels = None
        self.testData = None
        self.testLabels = None
        self.labelNames = None
        if download:
            self.download()

    def download(self):
        for (url, md5) in self.resources:
            filename = url.rpartition('/')[2]
            print(os.getcwd())
            if filename not in os.listdir(os.getcwd()):
                r = requests.get(url, allow_redirects=True)
                open(os.path.join(os.getcwd(), self.rootPath, filename), 'wb').write(r.content)

    def load(self):
        pass

    def get_label_names(self):
        return self.labelNames

    def get_train_data(self):
        return self.trainData

    def get_train_labels(self):
        return self.trainLabels

    def get_validation_data(self):
        return self.validationData

    def get_validation_labels(self):
        return self.validationLabels

    def get_test_data(self):
        return self.testData

    def get_test_labels(self):
        return self.testLabels
