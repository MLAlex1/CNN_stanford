import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    https://stats.stackexchange.com/questions/422114/the-correct-implementation-of-momentum-method-and-nag
    https://stats.stackexchange.com/questions/422239/update-rule-for-gradient-descent-with-momentum

    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)

    # andrew ng video implementation
    andrew_ng = True

    # matrix of zeros, same dim as w.
    # this velocity is initialised with zero.
    # NB : after that each velocity for each parameter will be different, as
    # we subtract lr * dw and dw is different for each parameter.
    v = config.get("velocity", np.zeros_like(w))
    if andrew_ng:
        # next_w = w - lr * (mu * v + (1 - mu) * dw ) = w - lr * mu * v - lr * dw + lr * mu * dw
        mu = config["momentum"]
        # v = mu * v + (1 - mu) * dw
        v = mu * v + dw # works better
        next_w = w - config["learning_rate"] * v

    else:
        # next_w = w - lr * dw + mu * v
        next_w = None
        ###########################################################################
        # TODO: Implement the momentum update formula. Store the updated value in #
        # the next_w variable. You should also use and update the velocity v.     #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # normally formula is (mu * v + lr * (1-mu) * dw), but in literature 1-mu is omitted.
        v = config["momentum"] * v - config["learning_rate"] * dw
        next_w = w + v

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    The advantage of using rmsprop over sgd with momentum is, as it divides by
    sqrt(cache) if will reduce big gradients (big number / big number) and increases
    small gradients (small number / small number). So in 2d if we have w and b coefficients
    (https://www.youtube.com/watch?v=_e-LFe_igno) where b is large updates (derivative) and w has Small
    updates (derivative), we would speed up small updates going closer to the minimum and reduce
    big updates so we don't overshoot

    ex : 0.001 / np.sqrt(0.001) = 0.03
         10 / np.sqrt(10) = 3.16

    so because we divide by sqrt(cache), what we effectively do is to change learning rate
    for parameters, that's why it's ADAPTIVE. (increase lr for small parameters, decrease for large)

    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    # again NB that cache is a matrix, so we do divide element-wise
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cache = config['cache']
    dr = config['decay_rate']
    cache =  dr * cache + (1 - dr) * dw ** 2
    next_w = w - config['learning_rate'] * dw / (np.sqrt(cache) + config['epsilon'])


    config['cache'] = cache
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Adam integrated sgd with momentum and rmsprop. We have the momentum variable
    given by :
    m = beta1 * m + (1 - beta1) * dw
    and adaptive learning rate method given by :
    v = beta2 * v + (1 - beta2) * dw**2


    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    beta1 = config['beta1']
    beta2 = config['beta2']
    eps = config['epsilon']
    lr = config['learning_rate']
    t = config['t'] + 1

    m = config['m'] * beta1 + (1 - beta1) * dw
    m_corr = m / (1 - beta1 ** t)

    v = config['v'] * beta2 + (1 - beta2) * dw**2
    v_corr = v / (1 - beta2 ** t)

    next_w = w - lr * m_corr / (np.sqrt(v_corr) + config['epsilon'])


    config['t'] = t
    config['v'] = v
    config['m'] = m
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
