# 残差网络（ResNet）

上一节介绍的批量归一化层对网络各层的中间输出做标准化，从而使训练时数值更加稳定。但对于深度模型来说，还有一个问题困扰着模型的训练。回忆[“正向传播、反向传播和计算图”](../chapter_deep-learning-basics/backprop.md)一节。在反向传播时，我们从输出层开始，朝着输入层方向逐层计算权重参数和中间变量的梯度。当我们将这些层串联在一起时，靠近输入层的权重梯度是该层与输出层之间每层中间变量梯度的乘积。假设这些梯度的绝对值小于1，靠近输入层的权重参数只得到很小的梯度，因此这些参数也很难被更新。

针对上述问题，ResNet增加了跨层的数据线路，从而允许中间变量的梯度快速向输入层传递，并更容易更新靠近输入层的模型参数 [1]。这一节我们将介绍ResNet的工作原理。


## 残差块

ResNet的基础块叫做残差块 (residual block) 。如图5.9所示，它将层A的输出在输入给层B的同时跨过B，并和B的输出相加作为下面层的输入。它可以看成是两个网络相加，一个网络只有层A，一个则有层A和B。这里层A在两个网络之间共享参数。在求梯度的时候，来自层B上层的梯度既可以通过层B也可以直接到达层A，从而让层A更容易获取足够大的梯度来进行模型更新。

![残差块（左）和它的分解（右）。](./resnet.svg)


ResNet沿用了VGG全$3\times 3$卷积层设计。残差块里首先是两个有同样输出通道的$3\times 3$卷积层，每个卷积层后跟一个批量归一化层和ReLU激活层。然后我们将输入跳过这两个卷积层后直接加在最后的ReLU激活层前。这样的设计要求两个卷积层的输出与输入形状一样，从而可以相加。如果想改变输出的通道数，我们需要引入一个额外的$1\times 1$卷积层来将输入变换成需要的形状后再相加。

残差块的实现如下。它可以设定输出通道数，是否使用额外的卷积层来修改输入通道数，以及卷积层的步幅大小。

```{.python .input  n=2}
import sys
sys.path.insert(0, '../../Week5/code/')

import gluonbook as gb
from mxnet import nd, gluon, init
from mxnet.gluon import nn

# 本类已保存在 gluonbook 包中方便以后使用。
class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)
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

查看输入输出形状一致的情况：

```{.python .input  n=3}
blk = Residual(3)
blk.initialize()
X = nd.random.uniform(shape=(4, 3, 6, 6))
blk(X).shape
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "(4, 3, 6, 6)"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

改变输出形状的同时减半输出高宽：

```{.python .input  n=4}
blk = Residual(6, use_1x1conv=True, strides=2)
blk.initialize()
blk(X).shape
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "(4, 6, 3, 3)"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## ResNet模型

ResNet前面两层跟前面介绍的GoogLeNet一样，在输出通道为64、步幅为2的$7\times 7$卷积层后接步幅为2的$3\times 3$的最大池化层。不同一点在于ResNet的每个卷积层后面增加的批量归一化层。

```{.python .input  n=5}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

GoogLeNet在后面接了四个由Inception块组成的模块。ResNet则是使用四个由残差块组成的模块，每个模块使用若干个同样输出通道的残差块。第一个模块的通道数同输入一致，同时因为之前已经使用了步幅为2的最大池化层，所以也不减小高宽。之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并减半高宽。

下面我们实现这个模块，注意我们对第一个模块块做了特别处理。

```{.python .input  n=6}
def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

接着我们为ResNet加入所有残差块。这里每个模块使用两个残差块。

```{.python .input  n=7}
net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))
```

最后与GoogLeNet一样我们加入全局平均池化层后接上全连接层输出。

```{.python .input  n=8}
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
```

这里每个模块里有4个卷积层（不计算$1\times 1$卷积层），加上最开始的卷积层和最后的全连接层，一共有18层。这个模型也通常被称之为ResNet-18。通过配置不同的通道数和模块里的残差块数我们可以得到不同的ResNet模型。

注意，每个残差块里我们都将输入直接加在输出上，少数几个通过简单的$1\times 1$卷积层后相加。这样一来，即使层数很多，损失函数的梯度也能很快的传递到靠近输入的层那里。这使得即使是很深的ResNet（例如ResNet-152），在收敛速度上也同浅的ResNet（例如这里实现的ResNet-18）类似。同时虽然它的主体架构上跟GoogLeNet类似，但ResNet结构更加简单，修改也更加方便。这些因素都导致了ResNet迅速被广泛使用。

最后我们考察输入在ResNet不同模块之间的变化。

```{.python .input  n=9}
X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "conv5 output shape:\t (1, 64, 112, 112)\nbatchnorm4 output shape:\t (1, 64, 112, 112)\nrelu0 output shape:\t (1, 64, 112, 112)\npool0 output shape:\t (1, 64, 56, 56)\nsequential1 output shape:\t (1, 64, 56, 56)\nsequential2 output shape:\t (1, 128, 28, 28)\nsequential3 output shape:\t (1, 256, 14, 14)\nsequential4 output shape:\t (1, 512, 7, 7)\npool1 output shape:\t (1, 512, 1, 1)\ndense0 output shape:\t (1, 10)\n"
 }
]
```

## 获取数据并训练

使用跟GoogLeNet一样的超参数，但减半了学习率。

```{.python .input  n=10}
import mxnet as mx
ctx = mx.gpu(1)

lr, num_epochs, batch_size = 0.05, 5, 256
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size, resize=96)
gb.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "training on gpu(1)\n"
 },
 {
  "ename": "MXNetError",
  "evalue": "[21:35:09] src/storage/./pooled_storage_manager.h:108: cudaMalloc failed: out of memory\n\nStack trace returned 10 entries:\n[bt] (0) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x31f81a) [0x7f621e07e81a]\n[bt] (1) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x31fe41) [0x7f621e07ee41]\n[bt] (2) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x29eea63) [0x7f622074da63]\n[bt] (3) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x29f3315) [0x7f6220752315]\n[bt] (4) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x459114) [0x7f621e1b8114]\n[bt] (5) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2508b0d) [0x7f6220267b0d]\n[bt] (6) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2508fe3) [0x7f6220267fe3]\n[bt] (7) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2480ff9) [0x7f62201dfff9]\n[bt] (8) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x248abf4) [0x7f62201e9bf4]\n[bt] (9) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x248ea93) [0x7f62201eda93]\n\n",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mMXNetError\u001b[0m                                Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-10-ead888919046>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0mtrainer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgluon\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mTrainer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnet\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcollect_params\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'sgd'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0;34m'learning_rate'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mlr\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[0mtrain_iter\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtest_iter\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgb\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mload_data_fashion_mnist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch_size\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mresize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m96\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 8\u001b[0;31m \u001b[0mgb\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtrain_ch5\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnet\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrain_iter\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtest_iter\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrainer\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_epochs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;32m~/CUHK_SZ/DL/MXnet/CUHK_SZ_DL/Week5/code/gluonbook/utils.py\u001b[0m in \u001b[0;36mtrain_ch5\u001b[0;34m(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)\u001b[0m\n\u001b[1;32m    696\u001b[0m             \u001b[0ml\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbackward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    697\u001b[0m             \u001b[0mtrainer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstep\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch_size\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 698\u001b[0;31m             \u001b[0mtrain_l_sum\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0ml\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masscalar\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    699\u001b[0m             \u001b[0mtrain_acc_sum\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0maccuracy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_hat\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    700\u001b[0m         \u001b[0mtest_acc\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mevaluate_accuracy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtest_iter\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnet\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mctx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/anaconda3/lib/python3.6/site-packages/mxnet/ndarray/ndarray.py\u001b[0m in \u001b[0;36masscalar\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   1892\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1893\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"The current array is not a scalar\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1894\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masnumpy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1895\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1896\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mastype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/anaconda3/lib/python3.6/site-packages/mxnet/ndarray/ndarray.py\u001b[0m in \u001b[0;36masnumpy\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   1874\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhandle\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1875\u001b[0m             \u001b[0mdata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mctypes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata_as\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mctypes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mc_void_p\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1876\u001b[0;31m             ctypes.c_size_t(data.size)))\n\u001b[0m\u001b[1;32m   1877\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1878\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/anaconda3/lib/python3.6/site-packages/mxnet/base.py\u001b[0m in \u001b[0;36mcheck_call\u001b[0;34m(ret)\u001b[0m\n\u001b[1;32m    147\u001b[0m     \"\"\"\n\u001b[1;32m    148\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mret\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 149\u001b[0;31m         \u001b[0;32mraise\u001b[0m \u001b[0mMXNetError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpy_str\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0m_LIB\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mMXGetLastError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    150\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    151\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mMXNetError\u001b[0m: [21:35:09] src/storage/./pooled_storage_manager.h:108: cudaMalloc failed: out of memory\n\nStack trace returned 10 entries:\n[bt] (0) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x31f81a) [0x7f621e07e81a]\n[bt] (1) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x31fe41) [0x7f621e07ee41]\n[bt] (2) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x29eea63) [0x7f622074da63]\n[bt] (3) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x29f3315) [0x7f6220752315]\n[bt] (4) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x459114) [0x7f621e1b8114]\n[bt] (5) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2508b0d) [0x7f6220267b0d]\n[bt] (6) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2508fe3) [0x7f6220267fe3]\n[bt] (7) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2480ff9) [0x7f62201dfff9]\n[bt] (8) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x248abf4) [0x7f62201e9bf4]\n[bt] (9) /home/zli/anaconda3/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x248ea93) [0x7f62201eda93]\n\n"
  ]
 }
]
```

## 小结

* 残差块通过将输入加在卷积层作用过的输出上来引入跨层通道。这使得即使非常深的网络也能很容易训练。

## 练习

- 参考ResNet论文的表1来实现不同版本的ResNet [1]。
- 对于比较深的网络， ResNet论文中介绍了一个“bottleneck”架构来降低模型复杂度。尝试实现它 [1]。
- 在ResNet的后续版本里，作者将残差块里的“卷积、批量归一化和激活”结构改成了“批量归一化、激活和卷积”，实现这个改进（[2]，图1）。

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1663)

![](../img/qr_resnet-gluon.svg)

## 参考文献

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings in deep residual networks. In European Conference on Computer Vision (pp. 630-645). Springer, Cham.
