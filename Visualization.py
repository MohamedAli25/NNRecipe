# Data Preprocessing Template

# Importing the libraries
import numpy as np
import mkl

import matplotlib.pyplot as plt
import pandas as pd

def plot1D(x,xlabel,ylabel,title="1d"):
       fig, ax = plt.subplots()
       ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
       ax.plot(range(0, len(x)), x)
       ax.grid()
       plt.show()
       fig.savefig(title+".png")


def plot2DGrayScale(image, h, w, title="2DGrayScale", need_reshape=False, normalized=True):

       #y = np.random.uniform(0,1,h*w)

       if need_reshape:
              image = np.reshape(image, (h, w))

       if not(normalized):
              image = image/256
       fig, ax = plt.subplots()
       ax.imshow(image, cmap=plt.cm.gray)
       ax.spines['left'].set_position(('outward', 10))
       ax.spines['bottom'].set_position(('outward', 10))

       # Hide the right and top spines
       ax.spines['right'].set_visible(False)
       ax.spines['top'].set_visible(False)

       # Only show ticks on the left and bottom spines
       ax.yaxis.set_ticks_position('left')
       ax.xaxis.set_ticks_position('bottom')
       plt.show()
       fig.savefig(title+".png")


def plot2DRGB(image, h, w, title="colored", normalized=True):

       #combined = np.random.uniform(0,1,(h,w,3))
       if not(normalized):
              image = image/256

       plot2DGrayScale(image[:, :, 0], h, w, "Red_Channel", False, True)
       plot2DGrayScale(image[:, :, 1], h, w, "Green_Channel", False, True)
       plot2DGrayScale(image[:, :, 2], h, w, "Blue_Channel", False, True)

       fig, ax = plt.subplots()

       ax.imshow(image) #draws RGB by default if the matrix size is (h,w,3)
       ax.spines['left'].set_position(('outward', 10))
       ax.spines['bottom'].set_position(('outward', 10))

       # Hide the right and top spines
       ax.spines['right'].set_visible(False)
       ax.spines['top'].set_visible(False)

       # Only show ticks on the left and bottom spines
       ax.yaxis.set_ticks_position('left')
       ax.xaxis.set_ticks_position('bottom')
       plt.show()
       fig.savefig(title+".png")

