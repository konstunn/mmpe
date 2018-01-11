""" This module tests Model from model """

import numpy as np
import tensorflow as tf
from model import Model
from importlib import reload

F = lambda th: [[th[0], 0.],
                [0., th[1]]]

C = lambda th: [[1.],
                [1.]]

G = lambda th: np.eye(2)
H = lambda th: [[1., 0.]]

X0 = lambda th: [[0],
                 [0]]
P0 = lambda th: 0.1 * np.eye(2)
Q = lambda th: 0.1 * np.eye(2)
R = lambda th: [[0.1]]

th = [-.5, -.5]

u = [2., 2.]

m = Model(F, C, G, H, X0, P0, Q, R, th)
