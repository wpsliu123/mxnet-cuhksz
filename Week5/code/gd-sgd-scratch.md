# Gradient descent and stochastic gradient descent from scratch

In the previous tutorials,
we decided *which direction* to move each parameter
and *how much* to move each parameter
by taking the gradient of the loss with respect to each parameter.
We also scaled each gradient by some learning rate, 
although we never really explained where this number comes from.
We then updated the parameters by performing a gradient step $\theta_{t+1} \gets \eta \nabla_{\theta}\mathcal{L}_t$. 
Each update is called a *gradient step*
and the process is called *gradient descent*.

The hope is that if we just take a whole lot of gradient steps,
we'll wind up with an awesome model that gets very low loss on our training data,
and that this performance might generalize to our hold-out data. 
But as a sharp reader, you might have any number of doubts. You might wonder, for instance: 

* Why does gradient descent work?
* Why doesn't the gradient descent algorithm get stuck on the way to a low loss?
* How should we choose a learning rate?
* Do all the parameters need to share the same learning rate?
* Is there anything we can do to speed up the process?
* Why does the solution of gradient descent over training data generalize well to test data?

Some answers to these questions are known. 
For other questions, we have some answers but only for simple models like logistic regression that are easy to analyze.
And for some of these questions, we know of best practices that seem to work even if they're not supported by any conclusive mathematical analysis. 
Optimization is a rich area of ongoing research. 
In this chapter, we'll address the parts that are most relevant for training neural networks. 
To begin, let's take a more formal look at gradient descent.

## Gradient descent in one dimension

To get going, consider a simple scenario in which we have one parameter to manipulate.
Let's also assume that our objective associates every value of this parameter with a value.
Formally, we can say that this objective function has the signature $f: \mathbb{R} \rightarrow \mathbb{R}$.
It maps from one real number to another.

Note that the domain of $f$ is in one-dimensional. According to its Taylor series expansion as shown in the [introduction chapter](./optimization-intro.ipynb), we have

$$f(x + \epsilon) \approx f(x) + f'(x) \epsilon.$$

Substituting $\epsilon$ with $-\eta f'(x)$ where $\eta$ is a constant, we have

$$f(x - \eta f'(x)) \approx f(x) -  \eta f'(x)^2.$$

If $\eta$ is set as a small positive value, we obtain

$$f(x - \eta f'(x)) \leq f(x).$$

In other words, updating $x$ as 

$$x := x - \eta f'(x)$$ 

may reduce the value of $f(x)$ if its current derivative value $f'(x) \neq 0$. Since the derivative $f'(x)$ is a special case of gradient in one-dimensional domain, the above update of $x$ is gradient descent in one-dimensional domain.

The positive scalar $\eta$ is called the learning rate or step size. Note that a larger learning rate increases the chance of overshooting the global minimum and oscillating. However, if the learning rate is too small, the convergence can be very slow. In practice, a proper learning rate is usually selected with experiments.

## Gradient descent  over multi-dimensional parameters

Consider the objective function $f: \mathbb{R}^d \rightarrow \mathbb{R}$ 
that takes any multi-dimensional vector $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$ as its input. 
The gradient of $f(\mathbf{x})$ with respect to $\mathbf{x}$ is defined by the vector of partial derivatives: 

$$\nabla_\mathbf{x} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

To keep our notation compact we may use the notation $\nabla f(\mathbf{x})$ and $\nabla_\mathbf{x} f(\mathbf{x})$ 
interchangeably when there is no ambiguity about which parameters we are optimizing over.
In plain English, each element $\partial f(\mathbf{x})/\partial x_i$ of the gradient 
indicates the rate of change for $f$ at the point $\mathbf{x}$ 
with respect to the input $x_i$ only. 
To measure the rate of change of $f$ in any direction 
that is represented by a unit vector $\mathbf{u}$, 
in multivariate calculus, we define the directional derivative of $f$ at $\mathbf{x}$ 
in the direction of $\mathbf{u}$ as

$$D_\mathbf{u} f(\mathbf{x}) = \lim_{h \rightarrow 0}  \frac{f(\mathbf{x} + h \mathbf{u}) - f(\mathbf{x})}{h},$$

which can be rewritten according to the chain rule as

$$D_\mathbf{u} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u}.$$

Since $D_\mathbf{u} f(\mathbf{x})$ gives the rates of change of $f$ at the point $\mathbf{x}$ 
in all possible directions, to minimize $f$, 
we are interested in finding the direction where $f$ can be reduced fastest. 
Thus, we can minimize the directional derivative $D_\mathbf{u} f(\mathbf{x})$ with respect to $\mathbf{u}$. 
Since $D_\mathbf{u} f(\mathbf{x}) = \|\nabla f(\mathbf{x})\| \cdot \|\mathbf{u}\|  \cdot \text{cos} (\theta) = \|\nabla f(\mathbf{x})\|  \cdot \text{cos} (\theta)$, 
where $\theta$ is the angle between $\nabla f(\mathbf{x})$ 
and $\mathbf{u}$, the minimum value of $\text{cos}(\theta)$ is -1 when $\theta = \pi$. 
Therefore, $D_\mathbf{u} f(\mathbf{x})$ is minimized 
when $\mathbf{u}$ is at the opposite direction of the gradient $\nabla f(\mathbf{x})$. 
Now we can iteratively reduce the value of $f$ with the following gradient descent update:

$$\mathbf{x} := \mathbf{x} - \eta \nabla f(\mathbf{x}),$$

where the positive scalar $\eta$ is called the learning rate or step size.

## Stochastic gradient descent

However, the gradient descent algorithm may be infeasible when the training data size is huge. Thus, a stochastic version of the algorithm is often used instead. 

To motivate the use of stochastic optimization algorithms, note that when training deep learning models, we often consider the objective function as a sum of a finite number of functions:

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}),$$

where $f_i(\mathbf{x})$ is a loss function based on the training data instance indexed by $i$. It is important to highlight that the per-iteration computational cost in gradient descent scales linearly with the training data set size $n$. Hence, when $n$ is huge, the per-iteration computational cost of gradient descent is very high.

In view of this, stochastic gradient descent offers a lighter-weight solution. At each iteration, rather than computing the gradient $\nabla f(\mathbf{x})$, stochastic gradient descent randomly samples $i$ at uniform and computes $\nabla f_i(\mathbf{x})$ instead. The insight is, stochastic gradient descent uses $\nabla f_i(\mathbf{x})$ as an unbiased estimator of $\nabla f(\mathbf{x})$ since

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$


In a generalized case, at each iteration a mini-batch $\mathcal{B}$ that consists of indices for training data instances may be sampled at uniform with replacement. 
Similarly, we can use 

$$\nabla f_\mathcal{B}(\mathbf{x}) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}\nabla f_i(\mathbf{x})$$ 

to update $\mathbf{x}$ as

$$\mathbf{x} := \mathbf{x} - \eta \nabla f_\mathcal{B}(\mathbf{x}),$$

where $|\mathcal{B}|$ denotes the cardinality of the mini-batch and the positive scalar $\eta$ is the learning rate or step size. Likewise, the mini-batch stochastic gradient $\nabla f_\mathcal{B}(\mathbf{x})$ is an unbiased estimator for the gradient $\nabla f(\mathbf{x})$:

$$\mathbb{E}_\mathcal{B} \nabla f_\mathcal{B}(\mathbf{x}) = \nabla f(\mathbf{x}).$$

This generalized stochastic algorithm is also called mini-batch stochastic gradient descent and we simply refer to them as stochastic gradient descent (as generalized). The per-iteration computational cost is $\mathcal{O}(|\mathcal{B}|)$. Thus, when the mini-batch size is small, the computational cost at each iteration is light.

There are other practical reasons that may make stochastic gradient descent more appealing than gradient descent. If the training data set has many redundant data instances, stochastic gradients may be so close to the true gradient $\nabla f(\mathbf{x})$ that a small number of iterations will find useful solutions to the optimization problem. In fact, when the training data set is large enough, stochastic gradient descent only requires a small number of iterations to find useful solutions such that the total computational cost is lower than that of gradient descent even for just one iteration. Besides, stochastic gradient descent can be considered as offering a regularization effect especially when the mini-batch size is small due to the randomness and noise in the mini-batch sampling. Moreover, certain hardware processes mini-batches of specific sizes more efficiently.

## Experiments

For demonstrating the aforementioned gradient-based optimization algorithms, we use the regression problem in the [linear regression chapter](../chapter02_supervised-learning/linear-regression-scratch.ipynb) as a case study.

```{.python .input  n=1}
# Mini-batch stochastic gradient descent.
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

```{.python .input  n=2}
import mxnet as mx
from mxnet import autograd
from mxnet import ndarray as nd
from mxnet import gluon
import random

mx.random.seed(1)
random.seed(1)

# Generate data.
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
X = nd.random_normal(scale=1, shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(scale=1, shape=y.shape)
dataset = gluon.data.ArrayDataset(X, y)

# Construct data iterator.
def data_iter(batch_size):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for batch_i, i in enumerate(range(0, num_examples, batch_size)):
        j = nd.array(idx[i: min(i + batch_size, num_examples)])
        yield batch_i, X.take(j), y.take(j)

# Initialize model parameters.
def init_params():
    w = nd.random_normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params

# Linear regression.
def net(X, w, b):
    return nd.dot(X, w) + b

# Loss function.
def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2
```

```{.json .output n=2}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zli/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.\n  from ._conv import register_converters as _register_converters\n"
 }
]
```

```{.python .input  n=3}
%matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt
import numpy as np

def train(batch_size, lr, epochs, period):
    assert period >= batch_size and period % batch_size == 0
    w, b = init_params()
    total_loss = [np.mean(square_loss(net(X, w, b), y).asnumpy())]
    # Epoch starts from 1.
    for epoch in range(1, epochs + 1):
        # Decay learning rate.
        if epoch > 2:
            lr *= 0.1
        for batch_i, data, label in data_iter(batch_size):
            with autograd.record():
                output = net(data, w, b)
                loss = square_loss(output, label)
            loss.backward()
            sgd([w, b], lr, batch_size)
            if batch_i * batch_size % period == 0:
                total_loss.append(
                    np.mean(square_loss(net(X, w, b), y).asnumpy()))
        print("Batch size %d, Learning rate %f, Epoch %d, loss %.4e" % 
              (batch_size, lr, epoch, total_loss[-1]))
    print('w:', np.reshape(w.asnumpy(), (1, -1)), 
          'b:', b.asnumpy()[0], '\n')
    x_axis = np.linspace(0, epochs, len(total_loss), endpoint=True)
    plt.semilogy(x_axis, total_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
```

```{.python .input  n=4}
train(batch_size=1, lr=0.2, epochs=3, period=10)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Batch size 1, Learning rate 0.200000, Epoch 1, loss 6.5111e-05\nBatch size 1, Learning rate 0.200000, Epoch 2, loss 8.4869e-05\nBatch size 1, Learning rate 0.020000, Epoch 3, loss 4.8941e-05\nw: [[ 2.0012236 -3.4004235]] b: 4.201426 \n\n"
 },
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

```{.python .input}
train(batch_size=1000, lr=0.999, epochs=3, period=1000)
```

```{.python .input}
train(batch_size=10, lr=0.2, epochs=3, period=10)
```

```{.python .input}
train(batch_size=10, lr=5, epochs=3, period=10)
```

```{.python .input}
train(batch_size=10, lr=0.002, epochs=3, period=10)
```

## Next
[Gradient descent and stochastic gradient descent with Gluon](../chapter06_optimization/gd-sgd-gluon.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
