from dataLoader import DataLoader

class MNISTDataLoader(DataLoader):
    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
         "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
         "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
         "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
         "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, path, download=False):
        pass
