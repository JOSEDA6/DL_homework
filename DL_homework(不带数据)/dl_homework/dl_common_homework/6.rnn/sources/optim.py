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
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    # ​**​*​**START OF YOUR CODE*​**​**
    # 更新速度向量：v = momentum * v - learning_rate * dw
    v = config["momentum"] * v - config["learning_rate"] * dw
    
    # 更新权重：w = w + v (动量项相当于积累了历史梯度方向)
    next_w = w + v
    # ​**​*​**END OF YOUR CODE*​**​**
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
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
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # ​**​*​**START OF YOUR CODE*​**​**
    # 获取缓存（梯度平方的指数移动平均）
    cache = config["cache"]
    
    # 更新梯度平方的指数移动平均：cache = decay_rate * cache + (1 - decay_rate) * (dw**2)
    cache = config["decay_rate"] * cache + (1 - config["decay_rate"]) * (dw**2)
    
    # 更新权重：w = w - learning_rate * dw / (sqrt(cache) + epsilon)
    next_w = w - config["learning_rate"] * dw / (np.sqrt(cache) + config["epsilon"])
    
    # 更新缓存
    config["cache"] = cache
    # ​**​*​**END OF YOUR CODE*​**​**

    return next_w, config


def adam(w, dw, config=None):
    """
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
    # ​**​*​**START OF YOUR CODE*​**​**
    # 获取Adam参数
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    m = config["m"]  # 一阶矩（动量）
    v = config["v"]  # 二阶矩（自适应学习率）
    t = config["t"] + 1  # 增加迭代计数（先更新t）
    config["t"] = t
    
    # 更新一阶矩估计：m = beta1 * m + (1 - beta1) * dw
    m = beta1 * m + (1 - beta1) * dw
    
    # 更新二阶矩估计：v = beta2 * v + (1 - beta2) * (dw**2)
    v = beta2 * v + (1 - beta2) * (dw**2)
    
    # 计算偏差校正后的一阶矩估计：m_hat = m / (1 - beta1**t)
    m_hat = m / (1 - beta1**t)
    
    # 计算偏差校正后的二阶矩估计：v_hat = v / (1 - beta2**t)
    v_hat = v / (1 - beta2**t)
    
    # 更新权重：w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    next_w = w - config["learning_rate"] * m_hat / (np.sqrt(v_hat) + config["epsilon"])
    
    # 更新中间变量
    config["m"] = m
    config["v"] = v
    # ​**​*​**END OF YOUR CODE*​**​**

    return next_w, config
