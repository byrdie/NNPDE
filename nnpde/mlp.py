
import numpy

import theano
import theano.tensor as T
from theano import pp

from hiddenLayer import HiddenLayer
from lossLayer import LossLayer

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, x, t, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        input = T.stack([x, t])

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.ip_layer_1 = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # No activation on last inner product layer
        self.ip_layer_2 = HiddenLayer(
            rng=rng,
            input=self.ip_layer_1.output,
            n_in=n_hidden,
            n_out=n_out,
            activation=None
        )

        self.output = self.ip_layer_2.output

        print pp(self.output)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.loss_layer = LossLayer(input=self.output, x=x, t=t)
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.ip_layer_1.W).sum()
            + abs(self.ip_layer_2.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.ip_layer_1.W ** 2).sum()
            + (self.ip_layer_2.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.loss = (
            self.loss_layer.loss
        )

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.ip_layer_1.params + self.ip_layer_2.params
        # end-snippet-3

        # keep track of model input
        self.input = input
