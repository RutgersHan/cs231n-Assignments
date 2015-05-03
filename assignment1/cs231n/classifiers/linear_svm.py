import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        t_margin = scores - correct_class_score + 1
        value_num = np.sum(t_margin>0) - 1  # subtract the corrent one, since it is alway 1>0
        dW[j] -= value_num * X[:,i].T
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[j] += X[:,i].T


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW /= num_train
  dW += reg * W


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
  pass
  scores = W.dot(X)
  num_classes = W.shape[0]
  num_train = X.shape[1]

  correct_class_score = scores[y, np.arange(num_train)]
  margin = scores - correct_class_score + 1
  margin[margin < 0] = 0
  margin[y, np.arange(num_train)] = 0
  loss = np.sum(margin) / num_train
  loss += 0.5 * reg * np.sum(W * W)


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
  pass
  margin_flag = margin>0;
  term2 = margin_flag.dot(X.T)

  margin_flag_sum = np.sum(margin_flag,axis=0)
  coeff_matrix = np.zeros((num_classes,num_train))
  coeff_matrix[y, np.arange(num_train)] = margin_flag_sum
  term1 = -coeff_matrix.dot(X.T)
  dW = term1 + term2
  dW = dW / num_train

  dW += reg * W



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
