# Batch Normalization in `gluon`

In the preceding section, [we implemented batch normalization ourselves](../chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.ipynb) using NDArray and autograd.
As with most commonly used neural network layers,
Gluon has batch normalization predefined,
so this section is going to be straightforward.

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)
#ctx = mx.cpu()
ctx = mx.gpu(1)
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

## The MNIST dataset

```{.python .input  n=2}
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

## Define a CNN with Batch Normalization

To add batchnormalization to a ``gluon`` model defined with Sequential,
we only need to add a few lines. 
Specifically, we just insert `BatchNorm` layers before the applying the ReLU activations.

```{.python .input  n=3}
num_fc = 512
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=5))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Activation(activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Activation(activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    
    # The Flatten layer collapses all axis, except the first one, into one axis.
    net.add(gluon.nn.Flatten())
    
    net.add(gluon.nn.Dense(num_fc))
    net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
    net.add(gluon.nn.Activation(activation='relu'))
    
    net.add(gluon.nn.Dense(num_outputs))
```

## Parameter initialization

```{.python .input  n=4}
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

## Softmax cross-entropy Loss

```{.python .input  n=5}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Optimizer

```{.python .input  n=6}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
```

## Write evaluation loop to calculate accuracy

```{.python .input  n=7}
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]
```

## Training Loop

```{.python .input}
epochs = 10
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) 
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
        
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))    
```

```{.json .output n=None}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.05262516087796785, Train_acc 0.98745, Test_acc 0.9844\nEpoch 1. Loss: 0.028981215642216413, Train_acc 0.9964833333333334, Test_acc 0.9933\nEpoch 2. Loss: 0.021661773114533107, Train_acc 0.9969333333333333, Test_acc 0.9929\nEpoch 3. Loss: 0.014967213743054866, Train_acc 0.9967666666666667, Test_acc 0.9903\nEpoch 4. Loss: 0.011788230125505828, Train_acc 0.9989833333333333, Test_acc 0.9932\n"
 }
]
```

## Next
[Introduction to recurrent neural networks](../chapter05_recurrent-neural-networks/simple-rnn.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
