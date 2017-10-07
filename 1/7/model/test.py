from model import Model
import numpy as np

F = lambda th: [[th[0], 1],
                [0, .5]]

C = lambda th: [[th[1]], [1]]

G = lambda th: np.eye(2)
H = lambda th: [1, 0]
X0 = lambda th: [0, -1]
P0 = lambda th: 0.1 * np.eye(2)
Q = lambda th: 0.1 * np.eye(2)
R = lambda th: 0.1
u = [2., 2.]
th = [.5, .5]

m = Model(F, C, G, H, X0, P0, Q, R, th)

# m.fim(u)

n = np.array(F(th)).shape[0]
s = len(th)
s2q = lambda s: int((s + 1) * s / 2 + 1)
q = s2q(s)
x = np.random.uniform(-1, 1, q*s).reshape([q, n])
p = [1/q] * q
