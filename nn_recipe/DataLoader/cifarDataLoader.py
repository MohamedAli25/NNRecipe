from dataLoader import *


class CifarDataLoader(DataLoader):
    _resources = [
        ('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
         'c58f30108f718f92721af3b95e74349a')
        # ('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 'eb9058c3a382ffc7106e4002c42a8d85'),
    ]

    def __init__(self, root_path, download=False):
        """ Loads the dataset and downloads it if required
        Args:
            root (str): The path to the dataset directory
            download (bool): A flag for downloading the dataset
                            (default is False)                      
        """
        os.chdir(root_path)
        for (url, md5) in self._resources:
            filename = url.rpartition('/')[2]
            if filename not in os.listdir(root_path) and download is True:
                urllib.request.urlretrieve(''.join((url, tar)), os.path.join(root_path, tar))

            with tarfile.open(os.path.join(root_path, filename)) as tar_object:
                members = [file for file in tar_object if file.name in files]
        return members

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load(self):

        pass

    def getData(self):
        pass

if __name__ == "__main__":
    cfDL = CifarDataLoader('E:\Engineering_courses\Senior\NN\Project', download=True)
    print(cfDL, type(cfDL))