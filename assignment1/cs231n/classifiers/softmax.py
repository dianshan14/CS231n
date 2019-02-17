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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
      scores = X[i].dot(W)
      # achieve numerical stability (avoid overflow)
      scores -= np.max(scores)

      # construct vector of softmax function for all classes
      softmax = np.exp(scores) / np.sum(np.exp(scores))
      loss += -np.log(softmax[y[i]])

      #############################################################################
      # The gradient of softmax loss L_i = -log(e^(f_{y_i}) / sum_{j} exp^(f_j)   #
      # is:                                                                       #
      #    dW = (o_{y_i} - 1) * X[i]     , if with respect to w_{y_i}             #
      #       = o_k                      , otherwise                              #
      #                                                                           #
      #    o : softmax function e^(f_{y_i}) / sum_{j} exp^(f_j)                   #
      #############################################################################
      for j in range(num_classes):
          if j == y[i]:
              dW[:, y[i]] += (softmax[y[i]]- 1) * X[i]
          else:
              dW[:, j] += softmax[j] * X[i]

  loss /= num_train
  dW /= num_train

  # reguliarzation loss
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  # achieve numerical stability (avoid overflow)
  scores -= np.max(scores, axis=1, keepdims=True)

  # construct softmax vector of softmax function for all classes
  # The parameter 'keepdims' in np.sum() is left in the result as dimensions with size one
  softmax = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
  loss = np.sum(-np.log(softmax[np.arange(num_train), y]))
  loss /= num_train
  loss += reg * np.sum(W * W)

  # Shallow explaination:
  # All X[i, j]*softmax[j] contribute to dW[: j], but have a exception.
  # All X[i, y[i]]*softmax[y[i]] should subtract contribution from one,
  # because its gradient's form (o_{y_i}-1)
  softmax[np.arange(num_train), y] += -1
  dW = X.T.dot(softmax)
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

