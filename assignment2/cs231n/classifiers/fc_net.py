from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        out1, lay1_cache = affine_forward(X, self.params['W1'], self.params['b1'])
        act1, act1_cache = relu_forward(out1)
        scores, scores_cache = affine_forward(act1, self.params['W2'], self.params['b2'])

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}

        # compute loss and backprop
        loss, dloss = softmax_loss(scores, y)
        dhidden, grads['W2'], grads['b2'] = affine_backward(dloss, scores_cache)
        dact1 = relu_backward(dhidden, act1_cache)
        _, grads['W1'], grads['b1'] = affine_backward(dact1, lay1_cache)

        # add regularization
        loss += 0.5 * self.reg * (np.sum(self.params['W1']*self.params['W1']) +
                                  np.sum(self.params['W2']*self.params['W2']))

        grads['W1'] += self.reg*self.params['W1']
        grads['W2'] += self.reg*self.params['W2']

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        dims = [input_dim] + hidden_dims + [num_classes]


        for i in range(self.num_layers):
            wtag = 'W{}'.format(i+1)
            btag = 'b{}'.format(i+1)

            self.params[wtag] = weight_scale * \
                    np.random.randn(dims[i], dims[i+1])
            self.params[btag] = np.zeros(dims[i+1])

        if self.normalization == 'batchnorm':
            for i in range(len(hidden_dims)):
                self.params['gamma'+'{}'.format(i+1)] = np.ones(dims[i+1])
                self.params['beta'+'{}'.format(i+1)] = np.zeros(dims[i+1])

        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        cache = {}
        layer_input = X
        for i in range(self.num_layers-1):

            wtag = 'W{}'.format(i+1)
            btag = 'b{}'.format(i+1)
            atag = 'ac{}'.format(i+1)
            dtag = 'd{}'.format(i+1)
            rtag = 'rc{}'.format(i+1)
            ntag = 'batch{}'.format(i+1)
            gtag = 'gamma{}'.format(i+1)
            betag = 'beta{}'.format(i+1)

            layer_input, cache[atag] = affine_forward(layer_input, \
                                                      self.params[wtag],
                                                      self.params[btag])

            layer_input, cache[rtag] = relu_forward(layer_input)
            if self.use_dropout:
                layer_input, cache[dtag] = dropout_forward(layer_input,
                                                           self.dropout_param)

            if self.normalization == 'batchnorm':
                layer_input, cache[ntag] = batchnorm_forward(layer_input,
                                                             self.params[gtag],
                                                             self.params[betag],
                                                             self.bn_params[i])

        scores, scores_cache = affine_forward(layer_input, \
                                        self.params['W{}'.format(self.num_layers)],
                                        self.params['b{}'.format(self.num_layers)])

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dloss = softmax_loss(scores, y)

        wtag = 'W{}'.format(self.num_layers)
        btag = 'b{}'.format(self.num_layers)

        upstream_grad, grads[wtag], grads[btag] = affine_backward(dloss, scores_cache)

        for i in range(1, self.num_layers)[::-1]:

            wtag = 'W{}'.format(i)
            btag = 'b{}'.format(i)
            atag = 'ac{}'.format(i)
            dtag = 'd{}'.format(i)
            rtag = 'rc{}'.format(i)
            ntag = 'batch{}'.format(i)
            gtag = 'gamma{}'.format(i)
            betag = 'beta{}'.format(i)

            if self.normalization == 'batchnorm':
                upstream_grad, grads[gtag], grads[betag] = \
                        batchnorm_backward_alt(upstream_grad, cache[ntag])
            if self.use_dropout:
                upstream_grad = dropout_backward(upstream_grad, cache[dtag])

            upstream_grad = relu_backward(upstream_grad, cache[rtag])
            upstream_grad, grads[wtag], grads[btag] = \
                affine_backward(upstream_grad, cache[atag])

            # add regularization
            loss += 0.5 * self.reg * np.sum(self.params[wtag]**2)
            grads[wtag] += self.reg*self.params[wtag]

        return loss, grads