import numpy as np
from src.NN.layers.linear import Linear
from src.NN.activations.activation import Identity, Sigmoid
from src.NN.losses.MeanSquared import MeanSquaredLoss
from src.opt.GD import GD

x = np.array([
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])
y = np.array([
    [-1],
    [1],
    [1],
    [-1]
])
opt = GD(learning_rate=0.1)
msl = MeanSquaredLoss()
l1 = Linear(in_dim=2, out_dim=2, activation=Identity())
l2 = Linear(in_dim=2, out_dim=1, activation=Identity())

for a in range(50):
    out = l2(l1(x))
    print("loss {}:".format(str(a)), msl(y, out))
    msl(y, out)
    delta = np.multiply( msl.local_grad.T, l2.local_grad["dZ"])
    opt.optimize(l2, delta)
    delta = np.dot(delta.T, l2.local_grad["dX"])

    delta = np.multiply(delta.T, l1.local_grad["dZ"])
    opt.optimize(l1, delta)

print(l1.weights)
print(l1.bias)
print(l2.weights)
print(l2.bias)
# print(l2(l1(np.array([[5,5]]))))

