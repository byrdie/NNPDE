'''
Define the PDE to be solved by NNPDE
'''
import theano
import theano.tensor as T

class PDE(object):
    def __init__(self):

        # Diffusion coefficient
        D = 1.0

        x = T.iscalar('x')
        t = T.iscalar('t')
        rho = T.iscalar('p')

        self.eq = D * T.grad(T.grad(rho, x)) - T.grad(rho,t)

        self.func = theano.function([rho, x,t], self.eq)

