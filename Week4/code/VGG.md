# Very deep networks with repeating elements

As we already noticed in AlexNet, the number of layers in networks keeps on increasing. This means that it becomes extremely tedious to write code that piles on one layer after the other manually. Fortunately, programming languages have a wonderful fix for this: subroutines and loops. This way we can express networks as *code*. Just like we would use a for loop to count from 1 to 10, we'll use code to combine layers. The first network that had this structure was VGG. 

## VGG

We begin with the usual import ritual

```{.python .input  n=1}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)
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

```{.python .input  n=2}
ctx = mx.gpu(1)
```

## Load up a dataset

```{.python .input  n=3}
batch_size = 64

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)
```

## The VGG architecture

A key aspect of VGG was to use many convolutional blocks with relatively narrow kernels, followed by a max-pooling step and to repeat this block multiple times. What is pretty neat about the code below is that we use functions to *return* network blocks. These are then combined to larger networks (e.g. in `vgg_stack`) and this allows us to construct VGG from components. What is particularly useful here is that we can use it to reparameterize the architecture simply by changing a few lines rather than adding and removing many lines of network definitions.

```{.python .input  n=4}
from mxnet.gluon import nn

def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(nn.Conv2D(channels=channels, kernel_size=3,
                      padding=1, activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out

def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out

num_outputs = 10
architecture = ((1,64), (1,128), (2,256), (2,512))
net = nn.Sequential()
with net.name_scope():
    net.add(vgg_stack(architecture))
    net.add(nn.Flatten())
    net.add(nn.Dense(512, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(512, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(num_outputs))
```

## Initialize parameters

```{.python .input  n=5}
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

## Optimizer

```{.python .input  n=6}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .05})
```

## Softmax cross-entropy loss

```{.python .input  n=7}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Evaluation loop

```{.python .input  n=8}
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]
```

## Training loop

```{.python .input}
###########################
#  Only one epoch so tests can run quickly, increase this variable to actually run
###########################
epochs = 10
smoothing_constant = .01

for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
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
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
        
        if i > 0 and i % 200 == 0:
            print('Batch %d. Loss: %f' % (i, moving_loss))
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))    
```

```{.json .output n=None}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Batch 200. Loss: 2.297618\nBatch 400. Loss: 2.166951\nBatch 600. Loss: 0.939590\nBatch 800. Loss: 0.393490\nEpoch 0. Loss: 0.2623796110016494, Train_acc 0.9398666666666666, Test_acc 0.9394\nBatch 200. Loss: 0.177964\nBatch 400. Loss: 0.150595\nBatch 600. Loss: 0.123598\nBatch 800. Loss: 0.104377\nEpoch 1. Loss: 0.10809132223002922, Train_acc 0.9428166666666666, Test_acc 0.9412\nBatch 200. Loss: 0.095668\n"
 }
]
```

## Next
[Batch normalization from scratch](../chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.ipynb)

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
