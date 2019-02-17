import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # Due to analytic gradient of SVM loss function, we get:
        # gradient (of loss) with respect to Wj
        dW[:, j] += X[i]
        # gradient (of loss) with respect to Wy_i
        dW[:, y[i]] += -X[i]
        # first dimension is total slice (D-dimension) and second dimension is one column indexing
        # so, we can update gradient along one axis

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Similarly, compute average gradient
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # the gradient of L2 regularization with respect to w_i_j is 2*strength*w_i_j
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  # size info: scores(500x10)
  scores = X.dot(W)

  # if use scores[:, y] will output (500, 500) matrix.
  # scores[np.arange(scores.shape[0]), y] use 'integer array indexing'
  # size info: y(500,), correct_scores(500,)
  correct_class_scores = scores[np.arange(scores.shape[0]), y]

  # compute SVM margin matrix
  # Note: delta is 1
  margins = scores - correct_class_scores.reshape(-1, 1) + 1
  # make correct class do not contribute to loss
  margins[np.arange(margins.shape[0]), y] = 0

  # sum margin and it be divided by size of examples(for computing average of all examples)
  loss = np.sum(margins[margins > 0]) / num_train
  # add regularization loss to data loss
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # compute contribution from every entry of 'margins'
  contribute = np.zeros(margins.shape)
  # for one sample, except for correct class, every class contribute to one gradient(X[i])
  contribute[margins > 0] = 1
  # *We sum all of incorrect class that could contribute to the loss.
  # In the early(during computing loss), we had excluded correct class,
  # so adding its contribution would not happen in the here.
  # For instance, C=10, we just compute the most 9 contribution in the here.
  contribute[np.arange(num_train), y] = -np.sum(contribute, axis=1)

  # *perfome dot product with X and contribute
  # Size info: X(500,3072), contribute(500,10), dW(3072,10)
  dW = X.T.dot(contribute)

  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
