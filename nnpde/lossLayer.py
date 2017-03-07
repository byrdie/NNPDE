
import numpy

import theano
import theano.tensor as T

from pde import PDE
from boundary import Boundary

class LossLayer(object):
    def __init__(self, input):
        self.input = input
        self.pde = PDE
        self.boundary = Boundary

    def loss(self, x, t):
        pde_diff = self.pde(self.input(x,t),x,t) ** 2
        bc_diff = (self.input(x, 0.0) - self.boundary(x)) ** 2
        return T.sqrt(pde_diff + bc_diff)