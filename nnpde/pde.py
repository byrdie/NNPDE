'''
Define the PDE to be solved by NNPDE
'''
import theano
import theano.tensor as T



class PDE(object):
    def __init__(self, input, x, t):

        # Diffusion coefficient
        D = 1.0

        self.eq = T.grad(T.grad(input, x), x) - T.grad(input, t)

        #self.t_func = theano.function([t], self.eq)

