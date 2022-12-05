from builtins import range
import numpy as np
import tqdm


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    Nbatches = x.shape[0]
    # if len(x) == 3:

    M = w.shape[1]

    out = (x.reshape(Nbatches, 1, -1).dot(w) + b).reshape(Nbatches, M)
    # else:
    #     out = x.reshape(Nbatches, -1).dot(w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dw = np.zeros(w.shape)
    dx = np.zeros(x.shape)
    db = np.zeros(b.shape)

    if len(x.shape) == 2:
        Nbatches = 1
    else:
        Nbatches = x.shape[0]

    # dX(X @ W) = G @ W.T
    # dW(X @ W) = X.T @ G
    if len(x.shape) == 2:
        dx += dout.dot(w.transpose()).reshape(x.shape[0],x.shape[1])
    elif len(x.shape) == 4:
        dx += dout.dot(w.transpose()).reshape(Nbatches,x.shape[1],x.shape[2], x.shape[3])
    else:
        dx += dout.dot(w.transpose()).reshape(Nbatches,x.shape[1],x.shape[2])
    dw += x.transpose().dot(dout).reshape(w.shape[0],w.shape[1], order="F")
    db += np.ones((1, dout.shape[0])).dot(dout)[0]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x.copy()
    out[x < 0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * np.where(cache > 0, 1, 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # num_train = x.shape[0]
    # scores = x.copy()
    # scores = scores - np.max(scores, axis=1, keepdims=True)
    #
    # # Softmax Loss
    # sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
    # softmax_matrix = np.exp(scores)/sum_exp_scores
    # loss = np.sum(-np.log(softmax_matrix[np.arange(num_train), y]) )
    #
    # # Gradient
    # softmax_matrix[np.arange(num_train), y] -= 1
    # # dW = X.T.dot(softmax_matrix)
    # dx = softmax_matrix.copy() / num_train
    # # Average
    # loss /= num_train
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    layernorm = bn_param.get('layernorm', 0)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        sample_mean = x.mean(0)
        sample_var = x.var(0)
        #
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
        #
        out = x_normalized * gamma + beta

        if layernorm == 0:
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = (x, x_normalized, gamma, sample_var, sample_mean, eps, layernorm)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_normalized = (x - running_mean) / np.sqrt(running_var)
        out = x_normalized * gamma + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, x_normalized, gamma, sample_var, sample_mean, eps = cache
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dgamma = np.sum(dout * x_normalized, 0)
    dbeta = np.sum(dout , 0)

    dldy = dout * gamma
    N = 1.0 * x.shape[0]
    dydv = -0.5 * ((sample_var + eps) ** (-1.5)) * (x - sample_mean)
    # dsdv = (sample_var + eps) ** (-0.5)
    dvdx = 2/N*(x- sample_mean)
    dydx = 1/np.sqrt(sample_var + eps)

    dydu = -1/np.sqrt(sample_var + eps)
    dvdu = -2/N * np.sum((x - sample_mean)   , axis=0)  

    dudx = 1/N

    # everywhere where x appears, so v, u and y -> all paths in the graph
    dx = np.sum(dldy*dydu, 0)*dudx + np.sum(dldy*dydv, 0) * (dvdx) + np.sum(dldy*dydv, 0) * dvdu*dudx + dldy * dydx

    # std = np.sqrt(sample_var + eps)
    # N = 1.0 * dout.shape[0]
    # dfdz = dout * gamma                              #[NxD]
    # dudx = 1/N                                                      #[NxD]
    # dvdx = 2/N * (x - sample_mean)                       #[NxD] 
    # dzdx = 1 / std                                      #[NxD]
    # dzdu = -1 / std                                     #[1xD]
    # dzdv = -0.5*((sample_var + eps)**-1.5)*(x - sample_mean)        #[NxD]
    # dvdu = -2/N * np.sum((x - sample_mean)   , axis=0)        #[1xD]

    # dx = dfdz*dzdx + np.sum(dfdz*dzdu,axis=0)*dudx + \
    #      np.sum(dfdz*dzdv,axis=0)*(dvdx+dvdu*dudx)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_normalized, gamma, sample_var, sample_mean, eps, layernorm = cache

    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - sample_mean) * (-1/2) * (sample_var + eps) ** (-3/2), 0)
    dmean = np.sum(dx_norm * (-1/np.sqrt(sample_var + eps)), 0) + dvar * np.mean(-2 * (x - sample_mean) ,0)
    dx = dx_norm * 1 / np.sqrt(sample_var + eps) + dvar * 2 * (x - sample_mean) / x.shape[0] + dmean / x.shape[0]


    dgamma = np.sum(dout * x_normalized, layernorm)
    dbeta = np.sum(dout , layernorm)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # you need to standardize over axis = 1, thus derivative also over axis = 1
    # and reshape as here
    sample_mean = x.mean(1).reshape(-1,1)
    sample_var = x.var(1).reshape(-1,1)

    # print(sample_mean)
    
    x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)

    out = x_normalized * gamma + beta


    cache = (x, x_normalized, gamma, sample_var, sample_mean, eps, 1)



    ln_param['mode'] = 'train' # same as batch norm in train mode
    ln_param['layernorm'] = 1
    # out, cache = batchnorm_forward(x.T, gamma.reshape(-1,1),
    #                                beta.reshape(-1,1), ln_param)
    # # transpose output to get original dims
    # out = out.T


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # dx, dgamma, dbeta = batchnorm_backward_alt(dout.T, cache)
    # transpose gradients w.r.t. input, x, to their original dims
    # dx = dx.T
    # dgamma = dgamma.T
    # dbeta = dbeta.T

    (x, x_normalized, gamma, sample_var, sample_mean, eps, layernorm) = cache
    # we divide by number of hidden units insted of batch size
    N = x.shape[1]

    dx_norm = (dout * gamma)
    # you need to standardize over axis = 1, thus derivative also over axis = 1
    # and reshape as in forward
    dvar = np.sum(dx_norm * (x - sample_mean) * (-1/2) * (sample_var + eps) ** (-3/2), 1)
    dmean = np.sum(dx_norm * (-1/np.sqrt(sample_var + eps)), 1) + dvar * np.mean(-2 * (x - sample_mean) ,1) 
    dvar = dvar.reshape(-1,1)
    dmean = dmean.reshape(-1,1)

    dx = dx_norm * 1 / np.sqrt(sample_var + eps) + dvar * 2 * (x - sample_mean) / N + dmean / N

    dgamma = np.sum(dout * x_normalized, 0)
    dbeta = np.sum(dout , 0)


    # dout = dout.T
    # dx_norm = (dout * gamma.T)
    # dvar = np.sum(dx_norm * (x.T - sample_mean.T) * (-1/2) * (sample_var.T + eps) ** (-3/2), 1)
    # dmean = np.sum(dx_norm * (-1/np.sqrt(sample_var.T + eps)), 1) + dvar * np.mean(-2 * (x.T - sample_mean.T) ,1) 

    # dx = dx_norm * 1 / np.sqrt(sample_var.T + eps) + dvar.reshape(-1,1) * 2 * (x.T - sample_mean.T) / x.shape[0] + dmean.reshape(-1,1) / x.shape[0]

    # dgamma = np.sum(dout.T * x_normalized, 1)
    # dbeta = np.sum(dout.T , 1)

    # dx = dx.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # print()
        mask = (np.random.rand(*x.shape) < p)/p
        out = x * mask

        # print(out)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    H1 = int(1 + (H + 2 * pad - HH)/stride)
    W1 = int(1 + (W + 2 * pad - WW)/stride)

    if pad != 0:
        padded_x = np.pad(x, [(0,), (0,), (pad,), (pad,)], 'constant')
    else:
        padded_x = x.copy()


    out = np.zeros((N, F, H1, W1))
    for n in range(N):
        for f in range(F):
            for i in range(H1):
                for j in range(W1):
                    out[n, f, i, j] = np.sum( padded_x[n, :, i*stride:i*stride+HH, j*stride : j*stride + WW] * w[f] ) + b[f]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive_old(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (x, w, b, conv_param) = cache

    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    H1 = int(1 + (H + 2 * pad - HH)/stride)
    W1 = int(1 + (W + 2 * pad - WW)/stride)

    padded_x = np.pad(x, [(0,), (0,), (pad,), (pad,)], 'constant')

    dx = np.zeros(x.shape)
    dx_second = np.zeros(x.shape)
    for n in range(N):
        # hi and wi you are looping through x
        for hi in range(H):
            for wi in range(W):
                for f in range(F):
                    relevant_weights = np.zeros_like(w[f, :, :, :])
                    for i in range(H1):
                        for j in range(W1):

                            mask_1 = np.zeros_like(w[f, :, :, :])
                            mask_2 = np.zeros_like(w[f, :, :, :])
                            
                            if ((hi + pad - i * stride) >= 0) and ((hi + pad - i * stride) < HH):
                                mask_1[:, hi + pad - i * stride, :] = 1.0
                                # y_idxs.append((hi + pad - i * stride))

                            if ((wi + pad - j * stride) >= 0) and ((wi + pad - j * stride) < WW):
                                mask_2[:, :, wi + pad - j * stride] = 1.0

                            relevant_weights += mask_1 * mask_2

                            w_masked = np.sum(w[f, :, :, :] * mask_1 * mask_2, axis=(1,2))

                            dx[n, :, hi, wi] += dout[n, f, i, j] * w_masked

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx_second, dw, db

def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (x, w, b, conv_param) = cache

    stride = conv_param['stride']
    pad = conv_param['pad']

    if pad != 0:
        padded_x = np.pad(x, [(0,), (0,), (pad,), (pad,)], 'constant')
    else:
        padded_x = x.copy()

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    H1 = int(1 + (H + 2 * pad - HH)/stride)
    W1 = int(1 + (W + 2 * pad - WW)/stride)

    dx = np.zeros(x.shape)
    for n in range(N):
        # hi and wi you are looping through x
        for hi in range(H):
            for wi in range(W):
                # i and j looking through output
                y_idxs = []
                w_idxs = []
                for i in range(H1):
                    for j in range(W1):
                        if ((hi + pad - i * stride) >= 0) and ((hi + pad - i * stride) < HH) and ((wi + pad - j * stride) >= 0) and ((wi + pad - j * stride) < WW):
                            w_idxs.append((hi + pad - i * stride, wi + pad - j * stride))
                            y_idxs.append((i, j))

                # loop through filters
                for f in range(F):
                    dx[n, : , hi, wi] += np.sum([w[f, :, widx[0], widx[1]] * dout[n, f, yidx[0], yidx[1]] for widx, yidx in zip(w_idxs, y_idxs)], 0)


    dw = np.zeros(w.shape)
    for f in range(F):
        for c in range(C):
            for i in range(HH):
                for j in range(WW):
                    dw[f, c, i ,j] += np.sum(padded_x[:,  c, i: i + H1 * stride : stride, j : j + W1* stride : stride] * dout[:, f, :, :])
                    # dw[f, c, i ,j] = np.sum(padded_x[:,  c, i: i + padded_x.shape[-2] - 1 : stride, j : j + padded_x.shape[-1] - 1 : stride] * dout[:, f, :, :])


    db = np.zeros((F))
    for f in range(F):
        db[f] = np.sum(dout[:, f, :, :])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    N, C, H, W = x.shape

    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)

    out = np.zeros((N, C, H_out, W_out))

    # for n in range(N):
    #     for c in range(C):
    #         for hi in range(0, H_out + 1, stride):
    #             for wi in range(0, W_out + 1, stride):
    #                 out[n, c, hi//stride, wi//stride] = np.max(x[n, c, hi : hi + pool_height, wi : wi + pool_width ]) 
    for n in range(N):
        for c in range(C):
            for hi in range(H_out):
                for wi in range(W_out):
                    out[n, c, hi, wi] = np.max(x[n, c, hi * stride : hi * stride + pool_height, wi * stride : wi * stride + pool_width ]) 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (x, pool_param) = cache

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    dx = np.zeros_like(x)
    N, C, H, W = x.shape

    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    i_t, j_t = np.where(np.max(x[n, c, i * stride : i * stride + pool_height, j * stride : j * stride + pool_width]) == x[n, c, i * stride : i * stride + pool_height, j * stride : j * stride + pool_width])
                    i_t, j_t = i_t[0], j_t[0]
                    dx[n, c, i * stride : i * stride + pool_height, j * stride : j * stride + pool_width][i_t, j_t] = dout[n, c, i, j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
    N, C, H, W = x.shape
    # as we standardize across batch dimension (N) as well as acrross features maps (H, W), we need first to transpose 
    # to use batchnorm_forward computing mean and std across axis 0, then
    # we need to reshape back to the original shape the outputs (0,2,3,1) - (N, H, W, C) and get it
    # back to (N, C, H, W) by transposing

    # so we basically normalize across individual features (C) as they might have different scales. Think of an
    # image with C = 3, red green and blue channels. So you just subtract mean divide by std for each of these red, 
    # green and blue channels as individual features might have different scales and we should account for that 
    # (this is not possible on feed forward layers, only in CNNs). 
    # The difference between batchnorm and instance norm is that here we compute these statistics for each feature
    # independently, BUT for all samples N. While instance norm computes it for one example at the time so we remove
    # batch dependence. Group norm goes further and computes still for 1 example, but takes multiple channels C into 
    # account to have more robust statistics. What you expect is that similar filters learn features within a similar 
    # scale, thus normalizing across these groups will produce more robust statistics. How do you make sure these groups
    # behave in this way? You hope through learning this happens, so you basically enforce this. 
    # Layer norm uses all channels to normalize, which is too restrictive. 

    x_resh = x.transpose(0,2,3,1).reshape(N*H*W, C)
    out, cache = batchnorm_forward(x_resh, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dout_resh = dout.transpose(0,2,3,1).reshape(N*H*W, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout_resh, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.

    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape

    x_resh = x.transpose(0, 2, 3, 1).reshape(G * N * H * W, C // G)

    sample_mean = x_resh.mean(1).reshape(-1,1)
    sample_var = x_resh.var(1).reshape(-1,1)

    x_normalized = (x_resh - sample_mean) / np.sqrt(sample_var + eps)
    x_normalized = x_normalized.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    
    out = x_normalized * gamma + beta
   
    cache = (x_resh, x_normalized, gamma, sample_var, sample_mean, eps, G)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (x, x_normalized, gamma, sample_var, sample_mean, eps, G) = cache
    # cache_batch = (x, x_normalized, gamma, sample_var, sample_mean, eps, 0)
    # we divide by number of hidden units insted of batch size

    N, C, H, W = dout.shape
    dout_resh = dout.transpose(0, 2, 3, 1).reshape(G * N * H * W, C // G)
    

    # dx_norm = (dout_resh * gamma)
    # print(dx_norm)

    # # you need to standardize over axis = 1, thus derivative also over axis = 1
    # # and reshape as in forward
    # dvar = np.sum(dx_norm * (x - sample_mean) * (-1/2) * (sample_var + eps) ** (-3/2), 1)
    # dmean = np.sum(dx_norm * (-1/np.sqrt(sample_var + eps)), 1) + dvar * np.mean(-2 * (x - sample_mean) ,1) 
    # dvar = dvar.reshape(-1,1)
    # dmean = dmean.reshape(-1,1)

    # dx = dx_norm * 1 / np.sqrt(sample_var + eps) + dvar * 2 * (x - sample_mean) / N + dmean / N

    dgamma = np.sum(dout * x_normalized.reshape(N, H, W, C).transpose(0, 3, 1, 2), axis=(0, 2,3)).reshape(1,C,1,1)
    dbeta = np.sum(dout , axis=(0, 2,3)).reshape(1,C,1,1)

    # dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
