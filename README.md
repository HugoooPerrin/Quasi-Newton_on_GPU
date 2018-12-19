# Quasi-Newton_on_GPU

Implementation of L-BFGS and VL-BFGS on GPU relying on PyTorch framework.

The VL variant is from [Large-scale L-BFGS using MapReduce](http://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce.pdf).

Here is an example of the speed-up (5 000 features and 200 000 samples):

![](https://github.com/HugoooPerrin/Quasi-Newton_on_GPU/blob/master/screen.png)
