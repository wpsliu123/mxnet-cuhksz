# This repository is mainly used for CUHK_SZ course CIE6032 and MDS6232: Selected Topics in Deep Learning

## On course tutorial

For on-course tutorial, we mainly based on the English version of mxnet tutorial [https://github.com/zackchase/mxnet-the-straight-dope] running on Ubuntu.

## Dependencies

Install Anaconda

```
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
chmod +x Anaconda3-5.2.0-Linux-x86_64.sh
./Anaconda3-5.2.0-Linux-x86_64.sh
```

Install Mxnet

If you have GPU installed on your laptop, please
```
pip install mxnet-cu90
```
(90 means that your cuda version is cuda 9.0, please check other version [https://pypi.org/project/mxnet-cu90/]).

Otherwise, please just install the CPU version of mxnet 
```
pip install mxnet
```

If you want to install notedown plug-in and open .md file in ipython notebook, please
```
pip install https://github.com/mli/notedown/tarball/master
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```
## Table of contents

### Week 1: Introduction of deep learning
### Week 2: Machine learning basics
### Week 3: Multilayer perceptron 
### Week 4: Convolutional neural networks (CNNs)
### Week 5: Optimization methods
### Week 6: Mid-term Quiz
### Week 7: Neural Network Structure
### Week 8-9: Neural Network Structure + Recurrent Neural Networks (RNNs)