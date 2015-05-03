import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights  # @Han, I believe this is K x N 
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[0]
  num_train = X.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  scores = W.dot(X)

  #numeric stability
  scores = scores - scores.max(0)
  exp_scores = np.exp(scores)
  sum_exp_scores = exp_scores.sum(0)
  normalized_exp_scores = exp_scores / (sum_exp_scores )
  
  for i in xrange(num_train):
    loss += -np.log(normalized_exp_scores[y[i],i] + 1e-5)
    for j in xrange(num_classes):
      if j==y[i]:
        dW[j] += (normalized_exp_scores[y[i],i] -1) * X[:,i]
      else:
        dW[j] += normalized_exp_scores[j,i] * X[:,i]

  dW = dW / num_train + reg * W
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)










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
  num_classes = W.shape[0]
  num_train = X.shape[1]
  coeff1 = np.zeros((num_classes,num_train))

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
 

  pass
  scores = W.dot(X)
  #numeric stability
  scores = scores - scores.max(0)
  exp_scores = np.exp(scores)
  sum_exp_scores = exp_scores.sum(0)
  normalized_scores = exp_scores / (sum_exp_scores)
  normalized_correct_scores = normalized_scores[y,np.arange(num_train)]
  log_cost = -np.log(normalized_correct_scores + 1e-5)
  loss =  np.sum(log_cost) / num_train + 0.5 * reg * np.sum(W * W) 
  

  coeff2 = normalized_scores.copy()
  coeff2[y,np.arange(num_train)] = 0


  dW_part2 = coeff2.dot(X.T)
  coeff1[y,np.arange(num_train)] = normalized_correct_scores - 1
  dW_part1 = coeff1.dot(X.T)
  dW =  (dW_part1 + dW_part2) / num_train + reg * W
  


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
