from nn_recipe.NN.ActivationFunctions import *
from nn_recipe.NN.LossFunctions import *
from nn_recipe.NN.Layers import *
from nn_recipe.Opt import *
from nn_recipe.NN import Network
import numpy as np
from nn_recipe.utility import OneHotEncoder
from nn_recipe.DataLoader.mnistDataLoader import MNISTDataLoader
import matplotlib.pyplot as plt
import os
from nn_recipe.NN.Layers.conv import Conv2D
from PIL import Image
from nn_recipe.NN.Layers.pooling import*
from nn_recipe.NN.Layers.flatten import Flatten

def main():
  
  net = Network(
      layers=[
          Linear(in_dim=784, out_dim=25, activation=ReLU()),
          Linear(in_dim=25, out_dim=10, activation=Identity())
      ],
      optimizer=GDAdaGrad(learning_rate=0.01),
      loss_function=MClassLogisticLoss(sum=True, axis=0),
  )

  # net.save("C:\\Users\\mgtmP\\Desktop\\mnist_net.net")
  mnist = MNISTDataLoader(rootPath="mnist", download=True)
  mnist.load()
  X = mnist.get_train_data().reshape((-1, 28*28))
  Y = mnist.get_train_labels().reshape((-1, 1))
  X_validate = mnist.get_validation_data().reshape((-1, 28*28))
  Y_validate = mnist.get_validation_labels().reshape((-1, 1))
  X_test = mnist.get_test_data().reshape((-1, 28*28))
  Y_test = mnist.get_test_labels().reshape((-1, 1))
  X = X / 255
  X_validate = X_validate / 255
  X_test = X_test / 255


  encoder = OneHotEncoder(
      types=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
      active_state=1,
      inactive_state=0
  )
  y_encoded = encoder.encode(Y)

  def verify():
    f = net.evaluate(X_validate)
    y_hat = encoder.decode(f)
    y_hat = np.array(y_hat).reshape((-1, 1))
    return np.count_nonzero(y_hat - Y_validate)

  net.train(X, y_encoded, notify_func=None, batch_size=100, max_itr=50, verify_func=verify)

  # net = Network.load("C:\\Users\\mgtmP\\Desktop\\mnist_net.net")
  out = net.evaluate(X_test)
  yhat = encoder.decode(out)
  yhat = np.array(yhat).reshape((-1, 1))
  print("Total Accuracy is :", 1-np.count_nonzero(yhat - Y_test)/Y_test.shape[0])
  plt.plot(net.errors)
  plt.xlabel('Number of Iterations')
  plt.ylabel('Number of Misclassified Examples')
  plt.show()
  
  """
  fltr = np.array([[0,0,0],[0,1,0],[0,0,0]]).reshape((1,3,3,1))
  sobel_fltr = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]).reshape((1,3,3,1))
  blur_fltr = np.ones((1,3,3,1))/9
  edge_fltr = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).reshape((1,3,3,1))
  #print((os.path.join('', 'andrew2.jpeg')))
  img = Image.open(os.path.join('', 'andrew2.jpeg')).convert("RGB")
  loss = MClassLogisticLoss()

  for epoch in range(10):
    print(f"epoch{epoch}")
    conv1 = Conv2D(inChannels=3, filters=3, filters_values=edge_fltr, padding="VALID")
    conv_out = conv1(np.array(img))
    p1 = MaxPool2D(kernelSize=3, strides=2, padding="SAME")
    p1_out = p1(conv_out)
    Image.fromarray(p1_out[0].astype(np.uint8) ,"RGB").show()
    flat = Flatten()
    f_out = flat(p1_out)
    b, _, cols = f_out.shape
    l1 = Linear(in_dim=cols, out_dim=3, activation=Sigmoid())
    out = l1(f_out[0])
    loss_val = loss(encoded_Y, out)
    print("loss", loss_val)
    delta = loss.local_grad
    delta = np.multiply(delta.T, l1.local_grad["dZ"])  # delta * ∂y/∂z
    gd.optimize(layer=l1, delta=delta)
    delta = np.dot(delta.T, l1.local_grad["dX"])
    delta = flat.calc_local_grad(delta)
    delta = p1.calc_local_grad(delta)
    delta = conv1.calc_local_grad(delta['dY'])
    gd.optimize(layer=conv1, delta=delta['dW'])
    """