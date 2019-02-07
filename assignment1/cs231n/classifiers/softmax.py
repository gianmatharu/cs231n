import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores) #for stability
    loss += -np.log(np.exp(scores[y[i]]) / np.sum(np.exp(scores)))

    p = lambda k:  np.exp(scores[k]) / np.sum(np.exp(scores))

    for j in range(num_classes):
        dW[:, j] += X[i] * p(j)
        if j == y[i]:
            dW[:, j] -= X[i]


  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)
  probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

  loss = np.sum(-np.log(probs[np.arange(num_train), y]))

  mask = np.zeros(probs.shape)
  mask[np.arange(num_train), y] += 1

  dW = np.matmul(X.T, (probs-mask))

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W

  return loss, dW

