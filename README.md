
Examples of automatic differentiation mentioned in the paper arXiv:~. The fold src contains utility functions
which are used to derive the numerical gradient by finite difference method.

* In the file "common_funcs.jl", we show the definition of the adjoint function for the non-holomorphic function "dot", and checked the gradient.

* In the file "arjovsky16.jl", we parameterize the W matrix using a mixture of real and complex numbers, which is used for unitary recurrent neural network in the paper [Unitary Evolution Recurrent Neural Networks](http://proceedings.mlr.press/v48/arjovsky16.pdf)

* In the file "wisdom.jl", we use complex numbers to parameterize the W matrix propose in the paper [Full-Capacity Unitary Recurrent Neural Networks](http://papers.nips.cc/paper/6327-full-capacity-unitary-recurrent-neural-networks). Compared to the original work which split the W matrix into two matrices which contain the real and imaginary parts separately, we use a single complex matrix to represent it, and the gradient of any complex function built with W can be computed as straightforward as the real case.


