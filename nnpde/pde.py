'''
Define the PDE to be solved by NNPDE
'''
import theano
import theano.tensor as T

x = T.iscalar('x')
t = T.iscalar('t')

class PDE(object):
    def __init__(self, input):

        # Diffusion coefficient
        D = 1.0

        self.eq = D * T.grad(T.grad(input[0], x), x) - T.grad(input[0],t)


        self.func = theano.function([x,t], self.eq)

