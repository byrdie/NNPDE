import numpy

import theano
import theano.tensor as T
from theano import pp

x = T.fscalar('x')
t = T.fscalar('t')

xv = T.fvector('X')
tv = T.fvector('T')

#rho = T.tanh(xv)
# pde = lambda i:T.grad(rho[i],xv[i])
#pde = T.grad(T.grad(rho,x), x) + T.grad(rho,t)


#r,updates = theano.map(fn=lambda i: T.grad(rho,xv), sequences=T.arange(xv.shape[0]))

#f = theano.function([xv], T.grad(rho, xv))

x_val = numpy.asarray([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.],theano.config.floatX)
t_val = numpy.asarray([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],theano.config.floatX)

test = numpy.transpose(numpy.stack([x_val,t_val]))

print(test)

#inp = f(xv,tv)

#print(inp)
