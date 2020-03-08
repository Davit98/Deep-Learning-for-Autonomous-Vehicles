import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg=0):
    """
    Softmax loss function, vectorized version.

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
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability.                         #
    #############################################################################
    x = np.dot(X,W)
    e_x = np.exp(x - np.max(x))
    apply_softmax = e_x / (e_x.sum(axis=1).reshape(-1,1))

    y_one_hot = np.zeros((y.size, y.max()+1))
    y_one_hot[np.arange(y.size),y] = 1

    input_clamp = np.maximum(1e-15, np.minimum(apply_softmax, 1 - 1e-15))
    loss = -np.sum(y_one_hot*np.log(input_clamp))/apply_softmax.shape[0]

    dloss_ce = -(y_one_hot/input_clamp)/apply_softmax.shape[0]
    
    dloss_sft = apply_softmax * (dloss_ce - np.sum(apply_softmax*dloss_ce,axis=1).reshape(x.shape[0],1))

    dW = X.T.dot(dloss_sft) 
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW