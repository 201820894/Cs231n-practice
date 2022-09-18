from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train): #각 training example에 대해서 
        scores = X[i].dot(W) #(1, C) vectoriczed implementation 생각 -> 각 class에 대한 score
        correct_class_score = scores[y[i]] #올바른 class의 score
        for j in range(num_classes):
            if j == y[i]: #for loop 에 해당하는 클래스랑 맞는애면 
                continue #자기 클래스는 svm에서 계산안함
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            #correct class score는 sample마다 나오는거
            
            if margin > 0: 
                loss += margin
                dW[:,j]+=X[i]/num_train
                dW[:,y[i]]-=X[i]/num_train
                # margin<0이면 loss가 0: correct class score가 score[j] 보다 1이상 큰거니까 잘 분류한거
                # 각 sample에 대한거니까, 한 sj에 영향 미치는애들은 X[i] 전체: 
                    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW+=2*W*reg
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores=X.dot(W)

    correct_scores=scores[np.arange(len(X)),y].reshape(-1,1) #둘다 numpy 배열로 전달해주면 되잖아 바보야
    
    dif=scores-correct_scores
    mask=(dif!=0) # 자기 자신 아닌애들만 1
    hinge_loss=np.maximum(scores-correct_scores+1, np.zeros_like(scores))*mask
    
    loss=np.sum(hinge_loss)/X.shape[0]+reg*np.sum(W*W)

    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
    hinge_loss[hinge_loss>0]=1
    
    valid_margin_count = hinge_loss.sum(axis=1)
    
    hinge_loss[np.arange(X.shape[0]), y] -= valid_margin_count #dout
    
    dW = (X.T).dot(hinge_loss) / X.shape[0]
    
    dW+=2*W*reg
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
