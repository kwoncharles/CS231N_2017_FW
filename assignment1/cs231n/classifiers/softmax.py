import numpy as np
from random import shuffle
from past.builtins import xrange

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
  
  length = X.shape[0]
  classes = W.shape[1]
  for i in xrange(length):
      result = X[i].dot(W)
      result -= np.max(result)
      expsum = np.sum(np.exp(result))
      f = lambda k:np.exp(result[k]) / expsum
      loss -= np.log(f(y[i]))
      '''
      d_sftm = np.exp(result) / expsum
      for j in range(classes):
          if j == y[i]:              # But, why? Why it subtract 1
              d_loss = d_sftm[j] - 1
          else:
              d_loss = d_sftm[j]
          dW[:,j] += d_loss * X[i]
          '''
      # SVM과 다르게 얼마나 틀렸는지를 측정
      for j in range(classes):
          if j == y[i]:
              # softmax는 score를 0~1사이로 regularize 해주므로 정답레이블의 점수에서 1을 뺀 것이
              # loss가 된다
              d_loss = f(j) - 1
          else:
              # 정답이 아닌 레이블은 점수 자체가 loss가 된다
              d_loss = f(j)
          dW[:,j] += d_loss * X[i]
    
  
  loss /= length
  loss += reg * np.sum(W*W)
  dW /= length
  dW += reg * W
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
  length = X.shape[0]
  classes = W.shape[1]
  
  result = X.dot(W)
  #max_idx = np.argmax(result,axis=1)
  #result[np.arange(length),max_idx] = 0
  #정답이 아닌 pred 레이블을 지워야한다.
  result = result- np.max(result, axis=1)[...,np.newaxis]
  expsum = np.sum(np.exp(result),axis=1,keepdims=True)
  sftm = np.exp(result)/expsum

  loss -= np.sum(np.log(sftm[np.arange(length),y]))
  
  # mask array to subtract 1 from correct labels score
  mask = np.zeros_like(sftm)
  mask[np.arange(length),y] = 1
  dW = X.T.dot(sftm - mask)
  
  dW /= length
  dW += reg*W
  loss /= length
  loss += reg*np.sum(W*W)
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

