'''
Define the initial conditions for the problem
'''
import theano
import theano.tensor as T


class Boundary(object):
    def __init__(self):

        # initial distribution coefficient
        tau = 10.0

        x = T.iscalar('x')

        self.eq = T.exp((-1.) * tau * x)

        self.func = theano.function([x], self.eq)