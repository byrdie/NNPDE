'''
Define the initial conditions for the problem
'''
import theano
import theano.tensor as T


class Boundary(object):
    def __init__(self, x):

        # initial distribution coefficient
        tau = 10.0


        self.eq = T.exp((-1.) * tau * x)

        self.func = theano.function([x], self.eq)