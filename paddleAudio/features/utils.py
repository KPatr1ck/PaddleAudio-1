import paddle
import numpy as np
def randint(high,use_paddle = True):
    if use_paddle:
        return int(paddle.randint(0,high=high))
    return int(np.random.randint(0,high=high))
def rand(use_paddle = True):
    if use_paddle:
        return float(paddle.rand((1,)))
    return float(np.random.rand(1))
def weighted_sampling(weights):
    n = len(weights)
    w = np.cumsum(weights)
    w = w/w[-1]
    flag = rand() < w
    return np.argwhere(flag)[0][0]
    
