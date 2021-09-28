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

## GD

Gradient descent refers to the general optimization methods of updating values of the model's parameters in the direction of the steepest descent.

GD requires all data to be loaded at the same time:

$$
L(W)=\frac{1}{N}\sum_{i=1}^{N}L_i(x_i,y_i,W)+\lambda R(W)\\
\nabla_WL(W)=\frac{1}{N}\sum_{i=1}^{N}\nabla_WL_i(x_i,y_i,W)+\lambda \nabla_WR(W)\\
$$

Full sum is expensive when $$N$$ is large.

Batch GD uses a minibatch of examples to approximate sum.

## SGD

Stochastic gradient descent approximate expectation via sampling:

$$
L(W)=\mathbb{E}_{(x,y) \sim p_{data}}[L(x,y,W)]+\lambda R(W)\\
\approx\frac{1}{N}\sum_{i=1}^{N}L(x_i,y_i,W)+\lambda R(W)\\
$$

$$
\nabla_WL(W)=\nabla_W\mathbb{E}_{(x,y) \sim p_{data}}[L(x,y,W)]+\lambda \nabla_WR(W)\\
\approx\frac{1}{N}\sum_{i=1}^{N}\nabla_WL_i(x_i,y_i,W)+\lambda \nabla_WR(W)\\
$$

We can prove that SGD is an **unbiased estimator** of the gradients.

> ###  Recap: Unbiased estimator, and why the definition of sample variance has (n-1) instead of n
> [https://en.wikipedia.org/wiki/Bias_of_an_estimator](https://en.wikipedia.org/wiki/Bias_of_an_estimator)
>
> Suppose $$X_1,...,X_n$$ are independent and identically distributed (i.i.d.) random variables with expectation $$\mu$$ and variance $$\sigma^2$$.
>
> Suppose that sample mean is $$ \overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_i $$ 
>
> and sample variance is $$ S^2 = \frac{1}{n}\sum_{i=1}^{n}(X_i-\overline{X})^2 $$
>
> - We know that $$\overline{X}$$, $$\mu$$ are constants, so $$\sum_{i=1}^{n}(\overline{X}-\mu)=n(\overline{X}-\mu)$$;
>
> - and $$\overline{X}-\mu=\frac{1}{n}\sum_{i=1}^{n}(X_i-\mu)$$;
> - and by [Bienaymé formula](https://en.wikipedia.org/wiki/Variance#Sum_of_uncorrelated_variables_.28Bienaym.C3.A9_formula.29), $$\mbox{Var}(\overline{X})=\frac{\sigma^2}{n}$$; the definition of variance is  $$\mbox{Var}(X)= E[(X-\mu)^2]$$.
>
> then
>
> 
> $$
> \begin{align*}
> E[S^2]&=E[\frac{1}{n}\sum_{i=1}^{n}(X_i-\overline{X})^2]\\
> &= E[\frac{1}{n}\sum_{i=1}^{n}((X_i-\mu) - (\overline{X} - \mu))^2] \\
> &= E[\frac{1}{n}\sum_{i=1}^{n}(X_i-\mu)^2 - \frac{2}{n}(\overline{X} - \mu)\sum_{i=1}^{n}(X_i-\mu) + \frac{1}{n}\sum_{i=1}^{n}(\overline{X}-\mu)^2] \\
> &= E[\frac{1}{n}\sum_{i=1}^{n}(X_i-\mu)^2 - \frac{2}{n}(\overline{X} - \mu)\sum_{i=1}^{n}(X_i-\mu) + (\overline{X}-\mu)^2] \\
> &= E[\frac{1}{n}\sum_{i=1}^{n}(X_i-\mu)^2 - \frac{2}{n}(\overline{X} - \mu)*n(\overline{X} - \mu) + (\overline{X}-\mu)^2] \\
> &= E[\frac{1}{n}\sum_{i=1}^{n}(X_i-\mu)^2 - (\overline{X}-\mu)^2] \\
> &= E[\frac{1}{n}\sum_{i=1}^{n}(X_i-\mu)^2] - E[(\overline{X}-\mu)^2] \\
> &= \sigma^2 - E[(\overline{X}-\mu)^2] = (1-\frac{1}{n})\sigma^2 < \sigma^2
> \end{align*}
> $$
> 
>
> Therefore, $$S^2$$ is a *biased estimator* of $$\sigma^2$$. So we must counter the bias by multiplying $$S^2$$ by $$\frac{n}{n-1}$$, and the *unbiased estimator* of $$\sigma^2$$ is  $$ S^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i-\overline{X})^2 $$.
>
> 
>
> 

Why SGD over GD: SGD is faster simply in the way that it descends more frequently than GD, therefore, faster. If a dataset has 10,000 examples, for GD it would be 10,000 examples per iteration (slow), while SGD with a batch size of 10 would be 1000x faster.

## SGD + Momentum

Problems with SGD

- High condition number: ratio of highest to smallest singular value of the Hessian matrix is large.

  > ### Recap: Hessian Matrix
  >
  > Suppose $$f:\mathbb{R}^n\to\mathbb{R}$$ is a function with vector input $$\textbf{x}\in \mathbb{R}^n$$ and scalar output $$f(\textbf{x}\in \mathbb{R})$$.
  >
  > If all second partial derivatives of $$f$$ exists and are continuous, then the Hessian matrix $$\textbf{H}$$ of $$f$$ is an $$n\times n$$ matrix:
  >
  > 
  > $$
  > \textbf{H}_f=
  > \begin{bmatrix}
  > \frac{\partial^2f}{\partial x_1^2} & \frac{\partial^2f}{\partial x_1 \partial x_2} & ... & \frac{\partial^2f}{\partial x_1 \partial x_n}\\
  > \frac{\partial^2f}{\partial x_2^2} & \frac{\partial^2f}{\partial x_2^2} & ... & \frac{\partial^2f}{\partial x_2 \partial x_n}\\
  > ...&...& &...\\
  > \frac{\partial^2f}{\partial x_n \partial x_1} & \frac{\partial^2f}{\partial x_n \partial x_2} & ... & \frac{\partial^2f}{\partial x_n^2}
  > \end{bmatrix}
  > $$
  > 
  >
  > or, $$(\textbf{H}_f)_{i,j}=\frac{\partial^2f}{\partial x_i \partial x_j}$$
  
  > ### Recap: Condition Number
  >
  > In the field of numerical analysis, the condition number of a function with respect to an argument measures how much the output value of the function can change for a small change in the input argument.

- Local minimum & saddle point: not convex.
- Stochastic part introduces noise.

Improvement: SGD + momentum.
