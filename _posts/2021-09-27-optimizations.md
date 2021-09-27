---

layout: post
title: "Optimizations"
author: "VGTHuang"
tags: cv ml

---

 Optimization aims to achieve

 $$
 w^*=\mbox{argmin}_w L(w)
 $$

 Where $$\omega$$ is the weights of a model and $$L$$ is the loss.

2 ways to compute gradient:

- **Numeric approach** - compute with definition of (partial) derivative

$$
\frac{\mbox{d}f(x)}{\mbox{d}x}=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
$$

- **Analytic approach** - compute $$\nabla_w L$$ in

$$
L=\frac{1}{N}\sum_{i=1}^{N}L_i\quad(+ \sum_{k}w^k )
$$


In practice it's always analytic approach, but numeric gradient could be used to check the correctness.

Checking with numeric gradient in PyTorch: `torch.autograd.gradcheck`

```python
from torch.autograd import gradcheck

class MyFunction(Function):
    # a customized function
    def forward(ctx, tensor, constant):
        # some forward computations
        return output
    
    def backward(ctx, grad_output):
        # some backward computations
        return grad_input, grad_weight, grad_bias

my_func = MyFunction.apply

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True))
test = gradcheck(my_func, input, eps=1e-6, atol=1e-4)
print(test)
```

For more info see [here](https://pytorch.org/docs/master/notes/extending.html).

### GD

Gradient descent refers to the general optimization methods of updating values of the model's parameters in the direction of the steepest descent.

### Batch GD

GD requires all data to be loaded at the same time:

$$
L(W)=\frac{1}{N}\sum_{i=1}^{N}L_i(x_i,y_i,W)+\lambda R(W)\\
\nabla_WL(W)=\frac{1}{N}\sum_{i=1}^{N}\nabla_WL_i(x_i,y_i,W)+\lambda \nabla_WR(W)\\
$$

Full sum is expensive when $$N$$ is large.

Batch GD uses a minibatch of examples to approximate sum.

### SGD

Stochastic gradient descent approximate expectation via sampling:

$$
L(W)=\mathbb{E}_{(x,y) \sim p_{data}}[L(x,y,W)]+\lambda R(W)\\
\approx\frac{1}{N}\sum_{i=1}^{N}L(x_i,y_i,W)+\lambda R(W)\\
$$

$$
\nabla_WL(W)=\nabla_W\mathbb{E}_{(x,y) \sim p_{data}}[L(x,y,W)]+\lambda \nabla_WR(W)\\
\approx\frac{1}{N}\sum_{i=1}^{N}\nabla_WL_i(x_i,y_i,W)+\lambda \nabla_WR(W)\\
$$

