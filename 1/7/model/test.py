from model import Model

F = lambda th: [[th[0]]]
C = lambda th: [[th[1]]]
G = lambda th: [[1]]
H = lambda th: [[1]]
X0 = lambda th: [0]
P0 = lambda th: [[0.1]]
Q = lambda th: [[0.1]]
R = lambda th: [[0.3]]
u = [2, 2]
th = [1, 1]

m = Model(F, C, G, H, X0, P0, Q, R, th)
# m.fim(u)
# 8.35  12.31
# 12.31 27.70
