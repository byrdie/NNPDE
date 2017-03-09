
import numpy as np

import theano
import theano.tensor as T
from theano import pp

class LossLayer(object):
    def __init__(self, input, x, t):
        self.input = input
        self.pred_boundary = theano.clone(self.input, {t: np.float32(0.0)})
        self.pde  = T.grad(T.grad(self.input, x), x) - T.grad(self.input, t)
        self.boundary = self.eq = T.exp((-1.)  * (x ** 2))

    def loss(self):
        pde_diff = self.pde ** 2
        bc_diff = (self.pred_boundary - self.boundary) ** 2
        return pde_diff + bc_diff