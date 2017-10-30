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

plan = m.rand_plan(u)
print(plan)
plan = plan[0]

dp = m.direct(plan, u, th)
print(dp)
dp = dp[0]
cp = m.clean(dp)
