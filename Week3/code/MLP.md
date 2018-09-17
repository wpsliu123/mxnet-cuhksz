# Multilayer perceptrons from scratch

In the previous chapters we showed how you could implement multiclass logistic regression 
(also called *softmax regression*)
for classifiying images of handwritten digits into the 10 possible categories
([from scratch](../chapter02_supervised-learning/softmax-regression-scratch.ipynb) and [with gluon](../chapter02_supervised-learning/softmax-regression-gluon.ipynb)). 
This is where things start to get fun.
We understand how to wrangle data, 
coerce our outputs into a valid probability distribution,
how to apply an appropriate loss function,
and how to optimize over our parameters.
Now that we've covered these preliminaries, 
we can extend our toolbox to include deep neural networks.

Recall that before, we mapped our inputs directly onto our outputs through a single linear transformation.
$$\hat{y} = \mbox{softmax}(W \boldsymbol{x} + b)$$

Graphically, we could depict the model like this, where the orange nodes represent inputs and the teal nodes on the top represent the output:
![](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/img/simple-softmax-net.png?raw=true)

If our labels really were related to our input data by an approximately linear function,
then this approach might be adequate.
*But linearity is a strong assumption*.
Linearity means that given an output of interest,
for each input,
increasing the value of the input should either drive the value of the output up
or drive it down,
irrespective of the value of the other inputs.

Imagine the case of classifying cats and dogs based on black and white images.
That's like saying that for each pixel, 
increasing its value either increases the probability that it depicts a dog or decreases it.
That's not reasonable. After all, the world contains both black dogs and black cats, and both white dogs and white cats. 

Teasing out what is depicted in an image generally requires allowing more complex relationships between
our inputs and outputs, considering the possibility that our pattern might be characterized by interactions among the many features. 
In these cases, linear models will have low accuracy. 
We can model a more general class of functions by incorporating one or more *hidden layers*.
The easiest way to do this is to stack a bunch of layers of neurons on top of each other.
Each layer feeds into the layer above it, until we generate an output.
This architecture is commonly called a "multilayer perceptron".
With an MLP, we're going to stack a bunch of layers on top of each other.

$$ h_1 = \phi(W_1\boldsymbol{x} + b_1) $$
$$ h_2 = \phi(W_2\boldsymbol{h_1} + b_2) $$
$$...$$
$$ h_n = \phi(W_n\boldsymbol{h_{n-1}} + b_n) $$

Note that each layer requires its own set of parameters.
For each hidden layer, we calculate its value by first applying a linear function 
to the activations of the layer below, and then applying an element-wise
nonlinear activation function. 
Here, we've denoted the activation function for the hidden layers as $\phi$.
Finally, given the topmost hidden layer, we'll generate an output.
Because we're still focusing on multiclass classification, we'll stick with the softmax activation in the output layer.

$$ \hat{y} = \mbox{softmax}(W_y \boldsymbol{h}_n + b_y)$$

Graphically, a multilayer perceptron could be depicted like this:

![](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/img/multilayer-perceptron.png?raw=true)

Multilayer perceptrons can account for complex interactions in the inputs because 
the hidden neurons depend on the values of each of the inputs. 
It's easy to design a hidden node that that does arbitrary computation,
such as, for instance, logical operations on its inputs.
And it's even widely known that multilayer perceptrons are universal approximators. 
That means that even for a single-hidden-layer neural network,
with enough nodes, and the right set of weights, it could model any function at all!
Actually learning that function is the hard part. 
And it turns out that we can approximate functions much more compactly if we use deeper (vs wider) neural networks.
We'll get more into the math in a subsequent chapter, but for now let's actually build an MLP.
In this example, we'll implement a multilayer perceptron with two hidden layers and one output layer.

## Imports

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
```

```{.json .output n=1}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/zli/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.\n  from ._conv import register_converters as _register_converters\n"
 }
]
```

## Set contexts

```{.python .input  n=2}
data_ctx = mx.cpu()
#model_ctx = mx.cpu()
model_ctx = mx.gpu(3)
```

## Load MNIST data

Let's go ahead and grab our data.

```{.python .input  n=3}
num_inputs = 784
num_outputs = 10
batch_size = 64
num_examples = 60000
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
train_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)
```

## Allocate parameters

```{.python .input  n=4}
#######################
#  Set some constants so it's easy to modify the network later
####################### 
num_hidden = 256
weight_scale = .01

#######################
#  Allocate parameters for the first hidden layer
####################### 
W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale, ctx=model_ctx)
b1 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)

#######################
#  Allocate parameters for the second hidden layer
####################### 
W2 = nd.random_normal(shape=(num_hidden, num_hidden), scale=weight_scale, ctx=model_ctx)
b2 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)

#######################
#  Allocate parameters for the output layer
####################### 
W3 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale, ctx=model_ctx)
b3 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=model_ctx)

params = [W1, b1, W2, b2, W3, b3]
```

Again, let's allocate space for each parameter's gradients.

```{.python .input  n=5}
for param in params:
    param.attach_grad()
```

## Activation functions

If we compose a multi-layer network but use only linear operations, then our entire network will still be a linear function. That's because $\hat{y} = X \cdot W_1 \cdot W_2 \cdot W_2 = X \cdot W_4 $ for $W_4 = W_1 \cdot W_2 \cdot W3$. To give our model the capacity to capture nonlinear functions, we'll need to interleave our linear operations with activation functions. In this case, we'll use the rectified linear unit (ReLU):

```{.python .input  n=6}
def relu(X):
    return nd.maximum(X, nd.zeros_like(X))
```

## Softmax output

As with multiclass logistic regression, we'll want the outputs to constitute a valid probability distribution. We'll use the same softmax activation function on our output to make sure that our outputs sum to one and are non-negative.

```{.python .input  n=7}
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / partition
```

## The *softmax* cross-entropy loss function

In the previous example, we calculated our model's output and then ran this output through the cross-entropy loss function:

```{.python .input  n=8}
def cross_entropy(yhat, y):
    return - nd.nansum(y * nd.log(yhat), axis=0, exclude=True)
```

Mathematically, that's a perfectly reasonable thing to do. However, computationally, things can get hairy. We'll revisit the issue at length in a chapter more dedicated to implementation and less interested in statistical modeling. But we're going to make a change here so we want to give you the gist of why.

Recall that the softmax function calculates $\hat y_j = \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}$, where $\hat y_j$ is the j-th element of the input ``yhat`` variable in function ``cross_entropy`` and $z_j$ is the j-th element of the input ``y_linear`` variable in function ``softmax``

If some of the $z_i$ are very large (i.e. very positive), $e^{z_i}$ might be larger than the largest number we can have for certain types of ``float`` (i.e. overflow). This would make the denominator (and/or numerator) ``inf`` and we get zero, or ``inf``, or ``nan`` for $\hat y_j$. In any case, we won't get a well-defined return value for ``cross_entropy``. This is the reason we subtract $\text{max}(z_i)$ from all $z_i$ first in ``softmax`` function. You can verify that this shifting in $z_i$ will not change the return value of ``softmax``.

After the above subtraction/ normalization step, it is possible that $z_j$ is very negative. Thus, $e^{z_j}$ will be very close to zero and might be rounded to zero due to finite precision (i.e underflow), which makes $\hat y_j$ zero and we get ``-inf`` for $\text{log}(\hat y_j)$. A few steps down the road in backpropagation, we starts to get horrific not-a-number (``nan``) results printed to screen.

Our salvation is that even though we're computing these exponential functions, we ultimately plan to take their log in the cross-entropy functions. It turns out that by combining these two operators ``softmax`` and ``cross_entropy`` together, we can elude the numerical stability issues that might otherwise plague us during backpropagation. As shown in the equation below, we avoided calculating $e^{z_j}$ but directly used $z_j$ due to $log(exp(\cdot))$.
$$\text{log}{(\hat y_j)} = \text{log}\left( \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}\right) = \text{log}{(e^{z_j})}-\text{log}{\left( \sum_{i=1}^{n} e^{z_i} \right)} = z_j -\text{log}{\left( \sum_{i=1}^{n} e^{z_i} \right)}$$

We'll want to keep the conventional softmax function handy in case we ever want to evaluate the probabilities output by our model. But instead of passing softmax probabilities into our new loss function, we'll just pass our ``yhat_linear`` and compute the softmax and its log all at once inside the softmax_cross_entropy loss function, which does smart things like the log-sum-exp trick ([see on Wikipedia](https://en.wikipedia.org/wiki/LogSumExp)).

```{.python .input  n=9}
def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)
```

## Define the model

Now we're ready to define our model

```{.python .input  n=10}
def net(X):
    #######################
    #  Compute the first hidden layer
    #######################
    h1_linear = nd.dot(X, W1) + b1
    h1 = relu(h1_linear)

    #######################
    #  Compute the second hidden layer
    #######################
    h2_linear = nd.dot(h1, W2) + b2
    h2 = relu(h2_linear)

    #######################
    #  Compute the output layer.
    #  We will omit the softmax function here
    #  because it will be applied
    #  in the softmax_cross_entropy loss
    #######################
    yhat_linear = nd.dot(h2, W3) + b3
    return yhat_linear
```

## Optimizer

```{.python .input  n=11}
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
```

## Evaluation metric

```{.python .input  n=12}
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()
```

## Execute the training loop

```{.python .input  n=13}
epochs = 10
learning_rate = .001
smoothing_constant = .01

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += nd.sum(loss).asscalar()


    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 1.2539560581684113, Train_acc 0.8825833, Test_acc 0.8818\nEpoch 1. Loss: 0.3326801812728246, Train_acc 0.92373335, Test_acc 0.9224\nEpoch 2. Loss: 0.2243192758599917, Train_acc 0.94888335, Test_acc 0.9489\nEpoch 3. Loss: 0.16586496216456095, Train_acc 0.96206665, Test_acc 0.9587\nEpoch 4. Loss: 0.12948063965241113, Train_acc 0.96885, Test_acc 0.964\nEpoch 5. Loss: 0.10605045728087426, Train_acc 0.9757, Test_acc 0.968\nEpoch 6. Loss: 0.08846965184013049, Train_acc 0.97865, Test_acc 0.9688\nEpoch 7. Loss: 0.07562802119155725, Train_acc 0.98155, Test_acc 0.9729\nEpoch 8. Loss: 0.0649310845370094, Train_acc 0.98506665, Test_acc 0.9736\nEpoch 9. Loss: 0.05596223647991816, Train_acc 0.98578334, Test_acc 0.9751\n"
 }
]
```

## Using the model for prediction
Let's pick a few random data points from the test set to visualize algonside our predictions. We already know quantitatively that the model is more accurate, but visualizing results is a good practice that can (1) help us to sanity check that our code is actually working and (2) provide intuition about what kinds of mistakes our model tends to make.

```{.python .input  n=14}
%matplotlib inline
import matplotlib.pyplot as plt

# Define the function to do prediction
def model_predict(net,data):
    output = net(data)
    return nd.argmax(output, axis=1)

samples = 10

mnist_test = mx.gluon.data.vision.MNIST(train=False, transform=transform)

# let's sample 10 random data points from the test set
sample_data = mx.gluon.data.DataLoader(mnist_test, samples, shuffle=True)
for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    im = nd.transpose(data,(1,0,2,3))
    im = nd.reshape(im,(28,10*28,1))
    imtiles = nd.tile(im, (1,1,3))
    
    plt.imshow(imtiles.asnumpy())
    plt.show()
    pred=model_predict(net,data.reshape((-1,784)))
    print('model predictions are:', pred)
    print('true labels :', label)
    break
```

```{.json .output n=14}
[
 {
  "data": {
   "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAABECAYAAACRbs5KAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvhp/UCwAAGgtJREFUeJztnXt0VNXZ/7+b3CkphnsI90sQXQiFgAZRQsCCUgw/BATb4ioKhksbcYFJhUBou4RWkDuyAgKFwq/UlBaxvGjVcCmiEIhccseXhISQC0iCkPuc7/vH5JzOZGbCZOZkhgz7s9azktlnz3me5+x9ntln344gCYlEIpG0fFq52wCJRCKR6IMM6BKJROIhyIAukUgkHoIM6BKJROIhyIAukUgkHoIM6BKJROIhOBXQhRAThBBZQogrQog4vYySSCQSSdMRjs5DF0J4AcgG8ByAAgBnAcwkma6feRKJRCKxF2da6CMAXCH5vyRrAPwVQJQ+ZkkkEomkqXg78d0QAPkmnwsAPNnYF4QQclmqRCKRNJ2bJDveL5MzAV1YSbMI2EKIuQDmOqFHIpFIHnby7MnkTEAvANDd5HM3AIUNM5FMBJAIyBa6RCKRNCfO9KGfBdBfCNFbCOELYAaAj/UxSyKRSCRNxeEWOsk6IcRCAJ8C8AKwk2SabpZJJBKJpEk4PG3RIWUtrMtlxIgRCAgIAACEh4djwIABNvN6e3ujW7duiI6OBgBkZWW5xEaJxN3MmDEDALB//34IIUASR44cQWlpKQDg9u3byMjIwOnTp3H58mV3mtpsJCQkWKQdO3YMK1aswPHjx60ebyLnSIbdL1OLCOi+vr4AgG7dumlpkyZNQu/evQEAI0eORFjYf309cOAAAGD27NmorKx0yNbXXnsNH3zwAby97XuIIYnr16/j+++/BwDs27cP7733nkO6JY4TGBiIqCjj7Nlly5ahf//+OH/+PBYvXozjx4/rpmfOnDkAgHbt2qFfv36YNWsWrl27hh07dgAAxo4diy+++ELL/9lnnwEAUlNTdbPhQUFt6Hz99dcoLi6Gj48PevXqZZGvoqICsbGx2Lp1q+42hISEoKCgAGVlZQgKCmo076pVq6AoCpYuXeq03oiICKxYsQIRERGN5jt27BgAYMyYMY6qsiugg6TLBMZZME0Sf39/fvnll/zyyy9pMBhoMBioKIr2f2Py/vvvN1mfKm+//TYVRbGQvLw8rl+/nmPHjjWTMWPGOKxLin3ys5/9zOaxwMBAhoeH88SJE1r519XVabJ3794m61u+fLnV9IEDB7K4uJjFxcVW64g1uX79Oq9fv+6S6xQTE8OysjKS5MWLF3nx4kV+9NFHmjz//PMcNWoUR40axfpGllPi5+dHPz8/9u3blwAYEBDAp556SpOFCxdyx44drKmpYVlZGcePH8/x48fr5m/Pnj25bds2GgwG3r17l71792bv3r1t5s/MzGRKSgpbt27ttG5bJCcnW013QleKXTH2QQ/o0dHRFoHa3oBeVlbGkJAQhy5gWloaFUVhaWkpS0tLOWTIEHp7e9PLy8slN2VD8fHx4cyZM1lRUcGKigp+/PHH7Ny5s0W+zp078+mnn+bgwYN11e/n58dx48Zx3LhxXL16Nc+cOUOSzM/P5+rVqy1k6dKlbNeuHdu1a8cf//jHutgQHBxsNb1169ZMSkrSgvfdu3d59+5d/vvf/9bSFi1a1GR9tm54Pz8/5uTkMCcnx+6Artaj5qof3t7e9Pb2ZkBAAOfNm8dbt27ZZddjjz3msjq8ePFiGgwG5ubmMjc3V7fzrlmzRrvna2trOXHiRE6cONEi3/Tp0zl9+nQtr636ZI9EREQwIiLCLHgnJydb5GtIQkKCozrtCuhycy6JRCLxFB7EFrppy+idd95pUgu9uLiY9+7d471791hZWckePXo49ItYW1vLuro6RkVFMSoqymWtGGvSt29f7tu3z8LXv/3tb3z77bf56aefMj09nenp6bx+/Tpv377N0tJSRkZG6mbD/Pnz7XoqsiYFBQWMjo5ututz+PBhrSX+wQcfMDg4mMHBwQwPD9fSR48erZu+l156ibW1taytrbW7hX7nzh3euXOHjz/+uK6+t2rVij/5yU94/PhxHj9+nIqi8OjRoywoKGBVVZVNe2pqalhTU8N+/fq5rB5PmzZN1xZ6dHQ0o6OjefPmTa2uNdYC3rdvn9l95EwL3VQSEhIYERFh85grW+gPVECPjIxkZGQk165dq6UNHTrUZkAvKipiUVERP//8c86YMYMzZsygv78/R4wYwREjRvCZZ55xuJBqa2tZWFjosspuS1577TXm5uZaDZS2ftjU9BdffFG3Cvv99987HNANBgN//vOf63pdOnfuzMOHD/Pw4cMkyZKSEm7dupUA2L17d3bv3p2HDx/WflD00tuvXz9mZ2fbHcgbSmFhoa5B/eWXX7bQMXToUM3W/Px85ufna8eys7M5adIktzRU1ICelZXFrKwsp8+n/oipdSwzM5Nt27a1mf/YsWM8duwYDQYD7927Z7XLUm9piK3Ab4fYFdCdWSmqOxMmTAAALFy4EFevXsXmzZtx4cIFLF++HADw2GOPAQBOnDiBb7/9FiUlJQCAq1evmp3nzJkzuthz8+ZNXc7jCK+88goAYMuWLfD19VV/EC24desWSktLkZOTAwAYPnw4goODdbXlmWeeQdu2bc3Szp07h9jYWISEhGi2AsZRfHVWkinZ2dm62TNz5kxs2LAB7dq1AwBcvHgREydOREFBAQAgNzcXAEASWVlZCA8P1033/Pnz0a9fP5vHT548CQAoKipCYGCgVqdVunTpghdeeAFpac4v2QgMDMSWLVugKAqSkpIAANeuXUNOTg68vb3xq1/9SpvxUVtbi40bN2LZsmWorq52WrcjREZGAvjvLDRnzzVy5Ejtc3V1NVasWIHy8nK7vv/ZZ5+huLjYaTtsERERgeTkZIt0dbZLs/EgtdDVEXmDwcDq6upmfUy3JV5eXtyzZw8VReHq1avZoUMHdujQgZMmTeLixYu5bds2pqam8vLly2aSlJTEbdu2ccyYMRwzZgx9fHwctqFnz57aoFvDlvitW7d469Ytbt++nSNHjmTPnj3Nvrtq1SpdW+hr1qzhnTt3zFrbMTEx7Nixo0XeuLg4VldXW9gaFRWly4wCABw1ahQvXLhgNoPF9NF5+fLlZrNc0tPTG221NUUSEhJYU1NjteW9e/duDho0iIGBgQwMDCRgHDx99tlnmZuba5Y3Pz+fAwYMcNqeoKAgKorC27dva91MgHGAdMOGDVQUhZmZmczMzOSoUaNcfi+ZysSJE1lTU8OcnBy2b9+e7du3d+p877//vsUEiFdeeYWPPPII27RpwzZt2pjl79+/vzahwGAw8MCBA7r6FxERYdG9YkpycrLVQdMmiBwUlUgkkoeKB6mFrv6Kqb+6+/btc3lLonv37lpL6tq1axYtsatXr/L27dtMSkqyKmqLKD4+3mEb/vnPf5rpLC8v59atWxsd4FVbaDdu3CBJ5uTkODwgbCofffSRRX/44sWLrebduXOnxWBoQUFBo3OCmyLDhg1jeXk56+rqeOjQIXbr1o3dunUjAHbt2pVbt241e0JQ8+kx1bRXr14sLy+3qA+JiYkMCwuzaBGayocffmi1Re+sTd7e3jx79iwVReHp06d5+vRpDh48mElJSVQUhenp6ezYsaPVpylXSlRUFDMyMmgwGLhmzRpdznnq1Cmb4zXq2Nr+/fv5xhtvcPTo0Vy9erVZngkTJujqY7KNeefWaGwQtRFpeYOir7/+Ol9//XXtoh87dszqfNLmlMjISKuP1J9//jnHjh3LgIAA/uhHP7L5/U6dOrFTp07cvXs34+Li6Ovr22Qb4uPjtYUrJ0+etCswT5s2TRt0UhRFl0HIPn368O7du6yurmZqaionT57MyZMnW/h09OhRHj16lHV1dVrZHThwgAEBAQwICHDajiFDhnDIkCFaQL1w4QKDg4O17o3ExESLQWG122nHjh1O6e7Zsyd79uzJzMxMs/qgBo0+ffrc9xzh4eEW9YnGG8JpGTx4MNPT0602PFwx6NeYLF++nMuXL2dVVRUNBgO3b9+u2zqOkydPOjVIv2TJEt38VOejN5Xk5GRtPrsdelpeQFdv0Hv37pm1ss6cOcMzZ85w0qRJzV4Jg4KCeOXKFVZUVHDnzp3aIoVWrVo16Txz586loii6rohrTFJSUpiSkkKDwcC0tLRGW4xNlV69etls5QUFBXHLli3csmWLVmbl5eXctGmTbvrVHyv1B2Pq1Kls3749T5w4oa0MNe1TN+1XdXZGSWxsLGNjYy0C5oQJE+xu5YWEhFhtJOh1fTp16sTU1FSmpqZq537rrbdcUu8AcNGiRVy0aJH2Q3Xu3DmzVbRZWVn86U9/qqvO/v3786uvvuJXX33Fb775hgUFBU0K6FevXuWmTZs4YMAArRHmqC2N9Z03JbjfR0/LC+iqtG7dmufOnbN6E5w/f1636Xi2pEePHuzVq5dT51AD+p///Odmv6FmzJihzYs2GAwcPnx4s+sEwA4dOjAtLc3iZhk3bpyuehoG9DVr1nDNmjVmAdxaQP/1r3/ttO60tDRt1bAqN2/e1Fru9pyjuQN6dHQ0f/jhB/7www+sqqrSBv+GDBniknqgNrgaDuDn5eUxLy/PqWBpr4SEhHDYsGEcNmwY4+LiGBcXx+3bt/Mf//hHo4G9rKyMGRkZzMjIcFj3/QK1tTRr6ffR03ICemhoqEULcNiwYdy7d6/2uGYqhYWF2lzbB1XUvWCc6Uu3R/z9/Xnu3Dmz6+MqH8PDw3njxg0z3fn5+Xz11Vd11dMwoJsG77q6OpaXl3Pfvn3aZ7Xv3tnuns6dO2sLcEwD8XvvvWf3OXx9fbWZR6biyN4y1qRr1668evUqq6qqWFVVxfHjx3PTpk1UFIVnz57V9llpznoQEhLCkJAQbe+WefPmsbCwUKsTv/jFL1xWJxuKEIKxsbFmdfTGjRu8fPkyMzIyeOfOHe3pJjQ01CEdap94QkKCJupnwLJLxtaMmPt0vchZLhKJRPJQ8SC00NevX88TJ04wKCiIQUFBZscGDRrExMREJiYmav1kiqIwLS3Nan53izpv/bvvvqOiKPztb3/brPomT55s1vI7fvy4y3zdvXu3xdOT6SpfvSQ0NJShoaHcu3cvFUVhZWWl9jifl5fHESNGMCEhQbNhzpw5nDNnjtN64+PjrQ6ONzYobiqDBw/mv/71L4tzFBcXc968ebpcm82bN1NRFE6ZMoVTpkwhYHwqyMjIoKIo2qppV9UJVRYsWKCVx8GDB12u31TU7iBVYmNjtev0xBNPuMQGdfBTbbVb4z7bArScLpf169fTYDBofXG2FoL4+/tz+/btWj9dTEwMY2JidNkCVBXTqXCOyJ49e7SFSaWlpc06ZczHx0fbm6KkpIQlJSUuq6Bjxoyx2A4gOztbtymKtmTq1KkWWxVHRkYyIyODdXV1zM/P102XtYCek5Nj9dG8a9eu2ratvXv3Znx8PPPy8iy+bzAYOHfuXF3sa9u2Le/du8f09HT6+/vT399fO6YO5Kr7l7iiTphK69attXpx/vx5l+tXZejQodrYkirz5893mz2A7Vkx9/leywnocXFxZhc8NTWVs2bNsjlTo+F855deekm3i33x4kVOmzatyd9r1aoV9+/fr/XjKorCpKSkZq0YK1as0K7Bxo0buXHjRpdUSC8vL7N9x1XZuXOnW26QzMxM7bqHhYXpdt4333zT6mBmTk4OP/zwQzMpKiqymled+aFuzvXLX/5SN/tiYmJYXV3N8PBwi2MLFy6koijalFJXl8nLL7+s1Yu//OUvbqkXADh8+HCLeqrXymFHJCIiwmxAVB0gtWPTrpazl8v69evRt29fzJ49GwDwxBNPYNeuXViyZAnWrVtn8dahCxcumH0ODQ112oYuXboAADp16oS33noLhw4dQk1NjV3f7dixI9555x3tVVwA8Kc//Qlr16512q7GmDJlCgAgLS1N2+/GFcTHx5vtj3Lp0iUAQFxcnMtsMEXduyY7O1vb00YP1q9fjxUrVgCA2V42ffv2Rd++fe06h6IoyMvLw+9//3sAwN69e3Wxzd/fH4sWLUJOTg5Onz5tdqx169Z48803AQAbNmxwSk/nzp3x7LPPwmAw4ODBg3Z/T62bAJCSkuKUDc4wbNgwl+iJiIiw+dai0aNH2zymvkVLh1fUAXDiJdF6UlVVhdWrV6NDhw4AgBdffBGAcTOu7du3W+RX31uoJ0VFRQCMry1LTExEQUGB9towU/Ly8lBRUYGBAwdqaVOnTtU2pPrd734HQL8CskVoaCgGDRoEkrhy5QrKysqaVZ8pTz/9NFq1+u94+p49ewBA2yzNFcyePVurG0IIXLp0CRMmTLB7cyZ7Ucvz3XffhZ+fn93fq6ysRFFREf7whz9g165dutoEAE8++SR69OiBN954w+JYYmIi+vTpg4KCAotg31S6d++OPXv2wMfHB1VVVQCMAfrvf/87Pv30U6ubrm3evBlTpkzB7du3AQCnTp1yygZnMH01ZXOSbGUjrvuxcuVK3TfrkrNcJBKJxFOwo9+7O4BkABkA0gDE1KcnALgO4Nt6ecHRPvSG8s033zS6/zZJs896DTIBxoGm+Ph47Z2Mpv2gpjTsIzUYDFy5ciW9vLxc8pq6//znPyTJ6upqzp49u9n1BQUFcdu2bdy2bRtrampoMBhYUVHBZcuWuXy/kNDQUJaUlJgtJJo6dWqz6gwLC+OSJUtYWlqqzfluKFu3buWSJUu4ZMmSZn9xxMGDB6koCseOHWuWPm7cOJaUlFBRFN32O58yZQpTUlK0F1OUlZVpi3JOnjzJpUuXapKVlcXq6mrm5+czPDzcav++q8TLy4tff/21S/rQExISrC4WUvvJG352214uAIIBDK3/PxBANoDHYAzoi/UYFLUmixcv5tmzZ60GdNPVaAaDQbetWU0lNDSUCxcu5MyZMzlz5ky+++67nD9/Pr/44guzYJ6Xl8f58+e7bHvSgQMHcuDAgbx79y4VRdFe6tDcEhwcrO0vo173O3fu8Mknn3SJflWef/55i+1zZ82a5fR2rC1N1NleakBX3/eqBvM//vGPTd6uwl7p378/FyxYwF27dvGTTz5hdnY2s7OzaTAYeOLECcbExLh14FEVX19fszhx6tQpnjp1yqH9lZoqpgFeHQh1MJCr0jyzXAAcAvAcmjmgA2CbNm04ffp0rlu3juvWrWNeXp7WMli3bh3DwsIYFhbWbBX3QZT9+/dz//792g/bqlWrml1nly5d+Mknn7CyspKVlZXaDXLkyBE+8sgjLvVffd1ceno6H330UT766KPa/uMPkzz++ONUFEVb6m86u+rQoUMP3PoMd4gQgr/5zW9YXFzMmzdvaitZ3W2Xg6J/QAfQC8A1AD+GMaDnArgIYCeAIL0DuhRLUeebqwG9vLy82YN6aGgoi4qKLJ6UtmzZ4vbr8bCKEIIrV66koig8cuQI165dy7Vr17Jjx466rsuQ8sCIvkv/hRBtAPwdwJsk7wD4AEBfAEMA3ABgdY6eEGKuECJFCOG+uUsSiUTyMGBny9wHwKcA3mqk5X5ZttCbXxYsWMAFCxawsLCQN27ccMmqN9MFTAaDgSdPnuRzzz3X7Js+SZEiRRO7WuiiPtDaRAghAPwZwPck3zRJDyZ5o/7/RQCeJDnDxmnU7zSuTCKRSCTWOEfyvpPq7QnoowCcBHAJgFKf/A6AmTB2txDGvvQ31ADfyLlKAdwDcPN+hnkIHfDw+ApIfz2dh8nfB83XniQ73i/TfQO63gghUuz5pfEEHiZfAemvp/Mw+dtSfZUrRSUSicRDkAFdIpFIPAR3BPREN+h0Fw+Tr4D019N5mPxtkb66vA9dIpFIJM2D7HKRSCQSD8FlAV0IMUEIkSWEuCKEcM+bEJoZIUSuEOKSEOJbdWWsEKKdEOLfQoic+r9B7rbTUYQQO4UQJUKIyyZpVv0TRjbWl/dFIcRQ91nuGDb8TRBCXK8v42+FEC+YHPttvb9ZQojx7rHaMYQQ3YUQyUKIDCFEmhAipj7dI8u3EX9bdvk2dXMuRwSAF4DvAPQB4AvgAoDHXKHblQLjfPwODdL+BCCu/v84AH90t51O+PcsgKEwWRVsyz8ALwD4HwACwFMAvnG3/Tr5mwArm9LBuAPpBQB+AHrX13cvd/vQBF9t7arqkeXbiL8tunxd1UIfAeAKyf8lWQPgrwCiXKTb3UTBuNIW9X8nu9EWpyB5AsD3DZJt+RcFYA+NfA3gESFEsGss1Qcb/toiCsBfSVaTvArgCoz1vkVA8gbJ8/X//wDj+w9C4KHl24i/tmgR5euqgB4CIN/kcwEav3gtFQL4TAhxTggxtz6tM+tX0Nb/7eQ265oHW/55cpkvrO9m2GnSheYx/gohegH4CYBv8BCUbwN/gRZcvq4K6MJKmidOr3ma5FAAzwNYIIR41t0GuRFPLXNbu4x6hL9WdlW1mdVKmif426LL11UBvQDGV9mpdANQ6CLdLoNkYf3fEgD/gPGRrFh9FK3/67o3KbsGW/55ZJmTLCZpIKkA2I7/Pna3eH+FED4wBrd9JA/WJ3ts+Vrzt6WXr6sC+lkA/YUQvYUQvgBmAPjYRbpdghDiR0KIQPV/AD8FcBlGP1+tz/YqjG988iRs+fcxgFn1syGeAlDO+2ze1hJo0E/8/2AsY8Do7wwhhJ8QojeA/gDOuNo+R6nfVfVDABkk3zc55JHla8vfFl++LhxVfgHGkeTvACx192hwM/jXB8ZR8Aswvkx7aX16ewBfAMip/9vO3bY64eP/h/ExtBbGFstrtvyD8RF1S315XwIQ5m77dfJ3b70/F2G8yYNN8i+t9zcLwPPutr+Jvo6CsQvhIkxe/O6p5duIvy26fOVKUYlEIvEQ5EpRiUQi8RBkQJdIJBIPQQZ0iUQi8RBkQJdIJBIPQQZ0iUQi8RBkQJdIJBIPQQZ0iUQi8RBkQJdIJBIP4f8AiUp2ZVmX0xcAAAAASUVORK5CYII=\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "model predictions are: \n[5. 5. 7. 7. 9. 0. 8. 3. 4. 3.]\n<NDArray 10 @gpu(3)>\ntrue labels : \n[5. 5. 7. 7. 9. 0. 8. 3. 4. 3.]\n<NDArray 10 @cpu(0)>\n"
 }
]
```

## Conclusion

Nice! With just two hidden layers containing 256 hidden nodes, respectively, we can achieve over 95% accuracy on this task. 

## Next
[Multilayer perceptrons with gluon](../chapter03_deep-neural-networks/mlp-gluon.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
