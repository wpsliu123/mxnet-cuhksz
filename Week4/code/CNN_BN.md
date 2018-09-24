# Batch Normalization from scratch

When you train a linear model, you update the weights
in order to optimize some objective.
And for the linear model, 
the distribution of the inputs stays the same throughout training.
So all we have to worry about is how to map 
from these well-behaved inputs to some appropriate outputs.
But if we focus on some layer in the middle of a deep neural network,
for example the third,
things look a bit different. 
After each training iteration, 
we update the weights in all the layers, including the first and the second.
That means that over the course of training,
as the weights for the first two layers are learned,
the inputs to the third layer might look dramatically different than they did at the beginning.
For starters, they might take values on a scale orders of magnitudes different from when we started training.
And this shift in feature scale might have serious implications, say for the ideal learning rate at each time. 

To explain, let us consider the Taylor's expansion for the objective function $f$ with respect to the updated parameter $\mathbf{w}$, such as $f(\mathbf{w} - \eta \nabla f(\mathbf{w}))$. Coefficients of those higher-order terms with respect to the learning rate $\eta$ may be so large in scale (usually due to many layers) that these terms cannot be ignored. However, the effect of common lower-order optimization algorithms, such as gradient descent, in iteratively reducing the objective function is based on an important assumption: all those higher-order terms with respect to the learning rate in the aforementioned Taylor's expansion are ignored.


Motivated by this sort of intuition, 
Sergey Ioffe and Christian Szegedy proposed [Batch Normalization](https://arxiv.org/abs/1502.03167),
a technique that normalizes the mean and variance of each of the features at every level of representation during training. 
The technique involves normalization of the features across the examples in each mini-batch.
While competing explanations for the technique's effect abound,
its success is hard to deny.
Empirically it appears to stabilize the gradient (less exploding or vanishing values)
and batch-normalized models appear to overfit less.
In fact, batch-normalized models seldom even use dropout. 
In this notebooks, we'll explain how it works.

## Import dependencies and grab the MNIST dataset
We'll get going by importing the typical packages and grabbing the MNIST data.

```python
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
mx.random.seed(1)
ctx = mx.gpu()
```

## The MNIST dataset

```python
batch_size = 64
num_inputs = 784
num_outputs = 10
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)
```

## Batch Normalization layer

The layer, unlike Dropout, is usually used **before** the activation layer 
(according to the authors' original paper), instead of after activation layer.

The basic idea is doing the normalization then applying a linear scale and shift to the mini-batch:

For input mini-batch $B = \{x_{1, ..., m}\}$, we want to learn the parameter $\gamma$ and $\beta$.
The output of the layer is $\{y_i = BN_{\gamma, \beta}(x_i)\}$, where:

$$\mu_B \leftarrow \frac{1}{m}\sum_{i = 1}^{m}x_i$$
$$\sigma_B^2 \leftarrow \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_B)^2$$
$$\hat{x_i} \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i \leftarrow \gamma \hat{x_i} + \beta \equiv \mbox{BN}_{\gamma,\beta}(x_i)$$

* formulas taken from Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." International Conference on Machine Learning. 2015.

With gluon, this is all actually implemented for us, 
but we'll do it this one time by ourselves,
using the formulas from the original paper
so you know how it works, and perhaps you can improve upon it!

Pay attention that, when it comes to (2D) CNN, we normalize `batch_size * height * width` over each channel.
So that `gamma` and `beta` have the lengths the same as `channel_count`.
In our implementation, we need to manually reshape `gamma` and `beta` 
so that they could (be automatically broadcast and) multipy the matrices in the desired way.

```python
def pure_batch_norm(X, gamma, beta, eps = 1e-5):
    if len(X.shape) not in (2, 4):
        raise ValueError('only supports dense or 2dconv')

    # dense
    if len(X.shape) == 2:
        # mini-batch mean
        mean = nd.mean(X, axis=0)
        # mini-batch variance
        variance = nd.mean((X - mean) ** 2, axis=0)
        # normalize
        X_hat = (X - mean) * 1.0 / nd.sqrt(variance + eps)
        # scale and shift
        out = gamma * X_hat + beta
    
    # 2d conv
    elif len(X.shape) == 4:
        # extract the dimensions
        N, C, H, W = X.shape
        # mini-batch mean
        mean = nd.mean(X, axis=(0, 2, 3))
        # mini-batch variance
        variance = nd.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
        # normalize
        X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        # scale and shift
        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))
    
    return out
```

Let's do some sanity checks. We expect each **column** of the input matrix to be normalized.

```python
A = nd.array([1,7,5,4,6,10], ctx=ctx).reshape((3,2))
A
```

```python
pure_batch_norm(A,
    gamma = nd.array([1,1], ctx=ctx), 
    beta=nd.array([0,0], ctx=ctx))
```

```python
ga = nd.array([1,1], ctx=ctx)
be = nd.array([0,0], ctx=ctx)

B = nd.array([1,6,5,7,4,3,2,5,6,3,2,4,5,3,2,5,6], ctx=ctx).reshape((2,2,2,2))
B
```

```python
pure_batch_norm(B, ga, be)
```

Our tests seem to support that we've done everything correctly.
Note that for batch normalization, implementing **backward** pass is a little bit tricky. 
Fortunately, you won't have to worry about that here, 
because the MXNet's `autograd` package can handle differentiation for us automatically.

Besides that, in the testing process, we want to use the mean and variance of the **complete dataset**, instead of those of **mini batches**. In the implementation, we use moving statistics as a trade off, because we don't want to or don't have the ability to compute the statistics of the complete dataset (in the second loop).

Then here comes another concern: we need to maintain the moving statistics **along with multiple runs of the BN**. It's an engineering issue rather than a deep/machine learning issue. On the one hand, the moving statistics are similar to `gamma` and `beta`; on the other hand, they are **not** updated by the gradient backwards. In this quick-and-dirty implementation, we use the global dictionary variables to store the statistics, in which each key is the name of the layer (`scope_name`), and the value is the statistics. (**Attention**: always be very careful if you have to use global variables!) Moreover, we have another parameter `is_training` to indicate whether we are doing training or testing.

Now we are ready to define our complete `batch_norm()`:

```python
def batch_norm(X,
               gamma,
               beta,
               momentum = 0.9,
               eps = 1e-5,
               scope_name = '',
               is_training = True,
               debug = False):
    """compute the batch norm """
    global _BN_MOVING_MEANS, _BN_MOVING_VARS
    
    #########################
    # the usual batch norm transformation
    #########################
    
    if len(X.shape) not in (2, 4):
        raise ValueError('the input data shape should be one of:\n' + 
                         'dense: (batch size, # of features)\n' + 
                         '2d conv: (batch size, # of features, height, width)'
                        )
    
    # dense
    if len(X.shape) == 2:
        # mini-batch mean
        mean = nd.mean(X, axis=0)
        # mini-batch variance
        variance = nd.mean((X - mean) ** 2, axis=0)
        # normalize
        if is_training:
            # while training, we normalize the data using its mean and variance
            X_hat = (X - mean) * 1.0 / nd.sqrt(variance + eps)
        else:
            # while testing, we normalize the data using the pre-computed mean and variance
            X_hat = (X - _BN_MOVING_MEANS[scope_name]) *1.0 / nd.sqrt(_BN_MOVING_VARS[scope_name] + eps)
        # scale and shift
        out = gamma * X_hat + beta
    
    # 2d conv
    elif len(X.shape) == 4:
        # extract the dimensions
        N, C, H, W = X.shape
        # mini-batch mean
        mean = nd.mean(X, axis=(0,2,3))
        # mini-batch variance
        variance = nd.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
        # normalize
        X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        if is_training:
            # while training, we normalize the data using its mean and variance
            X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        else:
            # while testing, we normalize the data using the pre-computed mean and variance
            X_hat = (X - _BN_MOVING_MEANS[scope_name].reshape((1, C, 1, 1))) * 1.0 \
                / nd.sqrt(_BN_MOVING_VARS[scope_name].reshape((1, C, 1, 1)) + eps)
        # scale and shift
        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))
      
    #########################
    # to keep the moving statistics
    #########################
    
    # init the attributes
    try: # to access them
        _BN_MOVING_MEANS, _BN_MOVING_VARS
    except: # error, create them
        _BN_MOVING_MEANS, _BN_MOVING_VARS = {}, {}
    
    # store the moving statistics by their scope_names, inplace    
    if scope_name not in _BN_MOVING_MEANS:
        _BN_MOVING_MEANS[scope_name] = mean
    else:
        _BN_MOVING_MEANS[scope_name] = _BN_MOVING_MEANS[scope_name] * momentum + mean * (1.0 - momentum)
    if scope_name not in _BN_MOVING_VARS:
        _BN_MOVING_VARS[scope_name] = variance
    else:
        _BN_MOVING_VARS[scope_name] = _BN_MOVING_VARS[scope_name] * momentum + variance * (1.0 - momentum)
        
    #########################
    # debug info
    #########################
    if debug:
        print('== info start ==')
        print('scope_name = {}'.format(scope_name))
        print('mean = {}'.format(mean))
        print('var = {}'.format(variance))
        print('_BN_MOVING_MEANS = {}'.format(_BN_MOVING_MEANS[scope_name]))
        print('_BN_MOVING_VARS = {}'.format(_BN_MOVING_VARS[scope_name]))
        print('output = {}'.format(out))
        print('== info end ==')
 
    #########################
    # return
    #########################
    return out
```

## Parameters and gradients

```python
#######################
#  Set the scale for weight initialization and choose 
#  the number of hidden units in the fully-connected layer 
####################### 
weight_scale = .01
num_fc = 128

W1 = nd.random_normal(shape=(20, 1, 3,3), scale=weight_scale, ctx=ctx) 
b1 = nd.random_normal(shape=20, scale=weight_scale, ctx=ctx)

gamma1 = nd.random_normal(shape=20, loc=1, scale=weight_scale, ctx=ctx)
beta1 = nd.random_normal(shape=20, scale=weight_scale, ctx=ctx)

W2 = nd.random_normal(shape=(50, 20, 5, 5), scale=weight_scale, ctx=ctx)
b2 = nd.random_normal(shape=50, scale=weight_scale, ctx=ctx)

gamma2 = nd.random_normal(shape=50, loc=1, scale=weight_scale, ctx=ctx)
beta2 = nd.random_normal(shape=50, scale=weight_scale, ctx=ctx)

W3 = nd.random_normal(shape=(800, num_fc), scale=weight_scale, ctx=ctx)
b3 = nd.random_normal(shape=num_fc, scale=weight_scale, ctx=ctx)

gamma3 = nd.random_normal(shape=num_fc, loc=1, scale=weight_scale, ctx=ctx)
beta3 = nd.random_normal(shape=num_fc, scale=weight_scale, ctx=ctx)

W4 = nd.random_normal(shape=(num_fc, num_outputs), scale=weight_scale, ctx=ctx)
b4 = nd.random_normal(shape=10, scale=weight_scale, ctx=ctx)

params = [W1, b1, gamma1, beta1, W2, b2, gamma2, beta2, W3, b3, gamma3, beta3, W4, b4]
```

```python
for param in params:
    param.attach_grad()
```

## Activation functions

```python
def relu(X):
    return nd.maximum(X, 0)
```

## Softmax output

```python
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition
```

## The *softmax* cross-entropy loss function

```python
def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)
```

## Define the model

We insert the BN layer right after each linear layer.

```python
def net(X, is_training = True, debug=False):
    ########################
    #  Define the computation of the first convolutional layer
    ########################
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=(3,3), num_filter=20)
    h1_normed = batch_norm(h1_conv, gamma1, beta1, scope_name='bn1', is_training=is_training)
    h1_activation = relu(h1_normed)
    h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h1 shape: %s" % (np.array(h1.shape)))
        
    ########################
    #  Define the computation of the second convolutional layer
    ########################
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=(5,5), num_filter=50)
    h2_normed = batch_norm(h2_conv, gamma2, beta2, scope_name='bn2', is_training=is_training)
    h2_activation = relu(h2_normed)
    h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h2 shape: %s" % (np.array(h2.shape)))
    
    ########################
    #  Flattening h2 so that we can feed it into a fully-connected layer
    ########################
    h2 = nd.flatten(h2)
    if debug:
        print("Flat h2 shape: %s" % (np.array(h2.shape)))
    
    ########################
    #  Define the computation of the third (fully-connected) layer
    ########################
    h3_linear = nd.dot(h2, W3) + b3
    h3_normed = batch_norm(h3_linear, gamma3, beta3, scope_name='bn3', is_training=is_training)
    h3 = relu(h3_normed)
    if debug:
        print("h3 shape: %s" % (np.array(h3.shape)))
        
    ########################
    #  Define the computation of the output layer
    ########################
    yhat_linear = nd.dot(h3, W4) + b4
    if debug:
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))
    
    return yhat_linear

```

## Test run

Can data be passed into the `net()`?

```python
for data, _ in train_data:
    data = data.as_in_context(ctx)
    break
```

```python
output = net(data, is_training=True, debug=True)
```

## Optimizer

```python
def SGD(params, lr):    
    for param in params:
        param[:] = param - lr * param.grad
```

## Evaluation metric

```python
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data, is_training=False) # attention here!
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()
```

## Execute the training loop

Note: you may want to use a gpu to run the code below. (And remember to set the `ctx = mx.gpu()` accordingly in the very beginning of this article.)

```python
epochs = 1
moving_loss = 0.
learning_rate = .001

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, num_outputs)
        with autograd.record():
            # we are in training process,
            # so we normalize the data using batch mean and variance
            output = net(data, is_training=True)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        if i == 0:
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy)) 
```

## Next
[Batch normalization with gluon](../chapter04_convolutional-neural-networks/cnn-batch-norm-gluon.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
