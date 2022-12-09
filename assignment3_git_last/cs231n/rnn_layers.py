"""This file defines layer types that are commonly used for recurrent neural networks.
"""

import numpy as np


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
    out = x.reshape(x.shape[0], -1).dot(w) + b
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
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


# def rnn_step_forward(x, prev_h, Wx, Wh, b):
#     """Run the forward pass for a single timestep of a vanilla RNN using a tanh activation function.

#     The input data has dimension D, the hidden state has dimension H,
#     and the minibatch is of size N.

#     Inputs:
#     - x: Input data for this timestep, of shape (N, D)
#     - prev_h: Hidden state from previous timestep, of shape (N, H)
#     - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
#     - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
#     - b: Biases of shape (H,)

#     Returns a tuple of:
#     - next_h: Next hidden state, of shape (N, H)
#     - cache: Tuple of values needed for the backward pass.
#     """
#     next_h, cache = None, None
#     ##############################################################################
#     # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
#     # hidden state and any values you need for the backward pass in the next_h   #
#     # and cache variables respectively.                                          #
#     ##############################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     next_h, _ = affine_forward(np.concatenate([prev_h, x], 1), np.concatenate([Wh, Wx]), b)
#     next_h_pretanh = next_h
#     next_h = np.tanh(next_h)
#     # cache = (np.concatenate([x, prev_h], 1), np.concatenate([Wx, Wh]), b)
#     cache = (x, prev_h, Wx, Wh, b, next_h)
#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     ##############################################################################
#     #                               END OF YOUR CODE                             #
#     ##############################################################################
#     return next_h, cache


# def rnn_step_backward(dnext_h, cache):
#     """Backward pass for a single timestep of a vanilla RNN.

#     Inputs:
#     - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
#     - cache: Cache object from the forward pass

#     Returns a tuple of:
#     - dx: Gradients of input data, of shape (N, D)
#     - dprev_h: Gradients of previous hidden state, of shape (N, H)
#     - dWx: Gradients of input-to-hidden weights, of shape (D, H)
#     - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
#     - db: Gradients of bias vector, of shape (H,)
#     """
#     dx, dprev_h, dWx, dWh, db = None, None, None, None, None
#     ##############################################################################
#     # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
#     #                                                                            #
#     # HINT: For the tanh function, you can compute the local derivative in terms #
#     # of the output value from tanh.                                             #
#     ##############################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     (x, prev_h, Wx, Wh, b, next_h) = cache

#     # 1 - tanh(x)**2 the derivative
#     dtanh = 1 - next_h**2

#     # dx = np.zeros_like(x)
#     # dx += (dnext_h * dtanh) @ Wx.T
#     #
#     # dprev_h = np.zeros_like(prev_h)
#     # dprev_h += (dnext_h * dtanh) @ Wh.T
#     #
#     # dWx = np.zeros_like(Wx)
#     # dWx += x.T @ (dnext_h * dtanh)

#     dx, dWx, db = affine_backward(dtanh * dnext_h, (x, Wx, b))
#     dprev_h, dWh, _ = affine_backward(dtanh * dnext_h, (prev_h, Wh, b))
#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     ##############################################################################
#     #                               END OF YOUR CODE                             #
#     ##############################################################################
#     return dx, dprev_h, dWx, dWh, db




def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Args:
        x: Input data for this timestep, of shape (N, D).
        prev_h: Hidden state from previous timestep, of shape (N, H)
        Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        b: Biases, of shape (H,)

    Returns a tuple of:
        next_h: Next hidden state, of shape (N, H)
        cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##########################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store next
    # hidden state and any values you need for the backward pass in the next_h
    # and cache variables respectively.
    ##########################################################################
    # Replace "pass" statement with your code
    
    next_z = x @ Wx + prev_h @ Wh + b
    next_h = np.tanh(next_z)
    cache = (x, Wx, Wh, next_h, prev_h)
    
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return next_h, cache


def rnn_forward(x, h0, Wx, Wh, b):
    """Run a vanilla RNN forward on an entire sequence of data.

    We assume an input sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the RNN forward,
    we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D)
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H)
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    all_cache = []
    N = x.shape[0]
    T = x.shape[1] # number of tokens
    h = np.zeros((N, T, h0.shape[1]))

    for t in range(T):
        if t == 0:
            next_h, cache = rnn_step_forward(x[:, t, :], h0, Wx, Wh, b)
            h[:, t , :] = next_h
            all_cache.append(cache)
        else:
            next_h, cache = rnn_step_forward(x[:, t, :], h[:, t-1 , :], Wx, Wh, b)
            h[:, t , :] = next_h
            all_cache.append(cache)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, all_cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Args:
        dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
        cache: Cache object from the forward pass

    Returns a tuple of:
        dx: Gradients of input data, of shape (N, D)
        dprev_h: Gradients of previous hidden state, of shape (N, H)
        dWx: Gradients of input-to-hidden weights, of shape (D, H)
        dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
        db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.
    #
    # HINT: For the tanh function, you can compute the local derivative in
    # terms of the output value from tanh.
    ##########################################################################
    # Replace "pass" statement with your code
    (x, Wx, Wh, next_h, prev_h) = cache

    dWx = x.T @ (dnext_h * (1 - next_h ** 2))
    dWh = prev_h.T @ (dnext_h * (1 - next_h ** 2))
    dprev_h = (dnext_h * (1 - next_h ** 2)) @ Wh.T
    dx = (dnext_h * (1 - next_h ** 2)) @ Wx.T
    db = (dnext_h * (1 - next_h ** 2)).sum(0)
    
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_backward(dh, cache):
    """Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    NOTE: 'dh' contains the upstream gradients produced by the
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, T, H = dh.shape
    D = cache[0][0].shape[-1]

    dx = np.zeros((N,T,D))
    dht = 0
    dWx = np.zeros((D, H))
    dWh = np.zeros((H,H))
    db = np.zeros(H)
    for t in range(T-1, -1, -1):
        dht += dh[:, t, :] # sum because dh gradient comes from previous h(t-1) and the output for h(t)
        dxt, dht, dWxt, dWht, dbt = rnn_step_backward(dht, cache[t])
        dWx += dWxt
        dWh += dWht
        db += dbt
        dx[:,t,:] = dxt
        
        # print(np.linalg.norm(dht))
        # print(np.linalg.norm(dWxt))

    dh0 = dht

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """Forward pass for word embeddings.

    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # (N, T) = x.shape
    # V, D = W.shape
    # em = np.zeros((N, T, V))
    # for i in range(N):
    #     em[i][np.arange(T), x[i]] = 1
    #     em[i][np.arange(T), x[i]] = 1
    #
    # x = (em @ W)
    # out = x

    out = W[x]
    cache = (x, W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """Backward pass for word embeddings.

    We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D)
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (x, W) = cache
    # dW = np.zeros_like(W)
    # dW = x.T @ dout
    dW = np.zeros_like(W)
    # it adds going through each dout row
    # values to dW in a row according to x indices
    np.add.at(dW, x, dout)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """A numerically stable version of the logistic sigmoid function."""
    pos_mask = x >= 0
    neg_mask = x < 0
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
    h = x @ Wx + prev_h @ Wh + b
    assert h.shape[-1] % 4 == 0
    ai, af, ao, ag = np.array_split(h, 4, axis=-1)
    i = sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = np.tanh(ag)
    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)
    
    cache = (x, prev_h, prev_c, Wx, Wh, h, np.tanh(next_c), i, f, o ,g)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    (x, prev_h, prev_c, Wx, Wh, next_h, next_c_t, i, f, o ,g) = cache
    
    dgrad = np.zeros((next_h.shape))
    assert dgrad.shape[1] % 4 == 0
    H = dgrad.shape[1] // 4
    
    # compute gradients wrt ai, af, ao and ag from two flows - next_h and next_c
    dnextc_dai = dnext_c * (i * (1-i)) * g
    dnextc_daf = dnext_c * (f * (1-f)) * prev_c
    dnextc_dao = 0
    dnextc_dag = dnext_c * (1 - g**2) * i
    
    dnexth_dai = dnext_h * o * (1 - next_c_t**2) * (i * (1-i)) * g
    dnexth_daf = dnext_h * o * (1 - next_c_t**2) * (f * (1-f)) * prev_c
    dnexth_dao = dnext_h * (o * (1-o) * next_c_t)
    dnexth_dag = dnext_h * o * (1 - next_c_t**2) * (1 - g**2) * i
    
    # join them together in a matrix at this point to conveniently compute
    # downstream gradients 
    dgrad[:, 0:H] = dnextc_dai + dnexth_dai
    dgrad[:, H:2*H] = dnextc_daf + dnexth_daf
    dgrad[:, 2*H:3*H] = dnextc_dao + dnexth_dao
    dgrad[:, 3*H:4*H] = dnextc_dag + dnexth_dag
    
    # now compute downstream gradients
    dx = dgrad @ Wx.T
    dprev_h = dgrad @ Wh.T
    dWx = x.T @ dgrad
    dWh = prev_h.T @ dgrad
    db = dgrad.sum(0)
    
    # we do dnext_h/dprev_c + dnext_c/dprev_c 
    dprev_c = dnext_c * f + dnext_h * o * (1 - next_c_t**2) * f
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """Forward pass for an LSTM over an entire sequence of data.

    We assume an input sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the LSTM forward,
    we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell state is set to zero.
    Also note that the cell state is not returned; it is an internal variable to the LSTM and is not
    accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    cache = []
    N, T, _ = x.shape
    H = h0.shape[-1]
    h = np.zeros((N, T, H))
    next_c = np.zeros((N, H))
    for t in range(x.shape[1]):
        xt = x[:, t , :]
        if t == 0:
            next_h, next_c, cache_s = lstm_step_forward(xt, h0, next_c, Wx, Wh, b)
            cache.append(cache_s)
        else:
            next_h, next_c, cache_s = lstm_step_forward(xt, next_h, next_c, Wx, Wh, b)   
            cache.append(cache_s)
    
        h[:, t, :] = next_h 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """Backward pass for an LSTM over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, T, H = dh.shape
    D = cache[0][0].shape[-1]
    dx = np.zeros((N, T, D))
    dWx = np.zeros((D, 4 * H))
    dWh = np.zeros((H, 4 * H))
    db = np.zeros((4 * H, ))
    dnext_c = np.zeros((N, H))
    for t in range(dh.shape[1]-1, -1, -1):
        if t == dh.shape[1] - 1:
            dx_s, dnext_h, dnext_c, dWx_s, dWh_s, db_s = lstm_step_backward(dh[:, t, :], dnext_c, cache[t])
        else:
            dx_s, dnext_h, dnext_c, dWx_s, dWh_s, db_s = lstm_step_backward(dh[:, t, :] + dnext_h, dnext_c, cache[t])
        dx[:, t, :] = dx_s
        dWx += dWx_s
        dWh += dWh_s
        db  += db_s

    
    dh0 = dnext_h

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """Forward pass for a temporal affine layer.

    The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """A temporal version of softmax loss for use in RNNs.

    We assume that we are making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores for all vocabulary
    elements at all timesteps, and y gives the indices of the ground-truth element at each timestep.
    We use a cross-entropy loss at each timestep, summing the loss over all timesteps and averaging
    across the minibatch.

    As an additional complication, we may want to ignore the model output at some timesteps, since
    sequences of different length may have been combined into a minibatch and padded with NULL
    tokens. The optional mask argument tells us which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose:
        print("dx_flat: ", dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
