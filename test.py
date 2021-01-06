class X:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        print(kwargs)

a = X()
a(mohamed="name")

n = Network(
    opt=GCD(beta1=4, beta2=1, learning_rate=0.1),

)

l = Layer(
    input_size=2,
    layer_size=4,
    (optional) activation=Identity(),
    (optional) weights_range=(max, min),
    (optional) weights_dist=Dist_Type
)

y = l(X)
l.weights = l.weights += 5

class Optimizer(ABC):
    def __init(self, learning_rate=0.1):
        self._lr = learning_rate

    @abstractMethod
    def optimize(layer:Layer, global_grad):
        pass

class SGD(Optimizer):
    def __init__(self):
        super().__init__()

    def optimize(layer:Layer, global_grad):
        pass


l0 = Layer(
    input_size=4,
    layer_size=3,
    activation=Sigmoid(),
)
l1 = Layer(
    input_size=3,
    layer_size=2,
    activation=Sigmoid(),
)
l2 = Layer(
    input_size=2,
    layer_size=1,
    activation=Sigmoid(),
)
opt = Optmizer()
y0 = l0(X)
y1 = l1(y0)
yhat = l2(y1)

global_grad = l2.local_grad
opt.optimize(l2, global_grad)
global_grad = calc_glob_grad(l1.local_grad, global_grad)
opt.optimize(l1, global_grad)
global_grad = calc_glob_grad(l0.local_grad, global_grad)
opt.optimize(l0, global_grad)

Sigmoid----
forward:
    a = 1/1+exp(-X)
    return a
local_grad:
    a(1-a)

X --> Z --> Y
X*dz = dx

Network faks
Layer       
Optimize    
