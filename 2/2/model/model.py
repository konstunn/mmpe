
import os
import math
import tensorflow as tf
import control
import autograd.numpy as np
import autograd
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
import scipy
import itertools
import copy
import operator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model(object):
    # TODO: introduce some more default argument values, check types, cast if
    # neccessary
    def __init__(self, F, C, G, H, x0_mean, x0_cov, w_cov, v_cov, th):
        """
        Arguments are all callables (functions) of 'th' returning python lists
        except for 'th' itself (of course)
        """

        # TODO: check if there are extra components in 'th'
        # TODO: evaluate and cast everything to numpy matrices first
        # TODO: cast scalars to numpy matrices
        # TODO: allow both constant matrices and callables

        def wrap_np(f):
            return lambda th: np.array(f(th), ndmin=2)

        self.__tf_F = F
        self.__tf_C = C
        self.__tf_G = G
        self.__tf_H = H
        self.__tf_x0_mean = x0_mean
        self.__tf_x0_cov = x0_cov
        self.__tf_w_cov = w_cov
        self.__tf_v_cov = v_cov

        # store arguments, after that check them
        F = self.__F = wrap_np(F)
        C = self.__C = wrap_np(C)
        G = self.__G = wrap_np(G)
        H = self.__H = wrap_np(H)
        x0_mean = self.__x0_mean = wrap_np(x0_mean)
        x0_cov = self.__x0_cov = wrap_np(x0_cov)
        w_cov = self.__w_cov = wrap_np(w_cov)
        v_cov = self.__v_cov = wrap_np(v_cov)

        th = self.__th = np.array(th, dtype=np.float64)

        # evaluate all functions
        F = F(th)
        C = C(th)
        H = H(th)
        G = G(th)
        w_cov = w_cov(th)    # Q
        v_cov = v_cov(th)    # R
        x0_m = x0_mean(th)
        x0_cov = x0_cov(th)  # P_0

        # get dimensions and store them as well
        self.__n = n = F.shape[0]
        self.__m = m = H.shape[0]
        self.__p = p = G.shape[1]
        self.__r = r = C.shape[1]

        x0_m = x0_m.reshape([n, 1])

        # generate means
        w_mean = np.zeros([p, 1], np.float64)
        v_mean = np.zeros([m, 1], np.float64)

        # and store them
        self.__w_mean = w_mean
        self.__v_mean = v_mean

        # check conformability
        u = np.ones([r, 1])
        # generate random vectors
        # squeeze, because mean must be one dimensional
        x = np.random.multivariate_normal(x0_m.flatten(), x0_cov)
        w = np.random.multivariate_normal(w_mean.flatten(), w_cov)
        v = np.random.multivariate_normal(v_mean.flatten(), v_cov)

        # shape them as column-vectors
        x = x.reshape([n, 1])
        w = w.reshape([p, 1])
        v = v.reshape([m, 1])

        # if model is not conformable, exception would be raised (thrown) here
        F @ x + C @ u + G @ w
        H @ x + v

        # check controllability, stability, observability
        # self.__validate()

        # if the execution reached here, all is fine so
        # define corresponding computational tensorflow graphs
        self.__define_observations_simulation()
        self.__define_likelihood_computation()

        self.__d_crit_to_opt_grad_f = autograd.grad(self.__d_crit_to_optimize)

    def __define_observations_simulation(self):
        # TODO: reduce code not to create extra operations

        self.__sim_graph = tf.Graph()
        sim_graph = self.__sim_graph

        r = self.__r
        m = self.__m
        n = self.__n
        p = self.__p
        s = len(self.__th)

        x0_mean = self.__tf_x0_mean
        x0_cov = self.__tf_x0_cov

        with sim_graph.as_default():

            # FIXME: shape must be [1 x s]
            th = tf.placeholder(tf.float64, shape=(s), name='th')

            # TODO: this should be continuous function of time
            # but try to let pass array also
            u = tf.placeholder(tf.float64, shape=[r, None], name='u')

            t = tf.placeholder(tf.float64, shape=[None], name='t')

            # TODO: refactor

            # FIXME: gradient of py_func is None
            # TODO: embed function itself in the graph, must rebuild the graph
            # if the structure of the model change
            # use tf.convert_to_tensor
            F = tf.convert_to_tensor(self.__tf_F(th), tf.float64)
            F.set_shape([n, n])

            C = tf.convert_to_tensor(self.__tf_C(th), tf.float64)
            C.set_shape([n, r])

            G = tf.convert_to_tensor(self.__tf_G(th), tf.float64)
            G.set_shape([n, p])

            H = tf.convert_to_tensor(self.__tf_H(th), tf.float64)
            H.set_shape([m, n])

            x0_mean = tf.convert_to_tensor(x0_mean(th), tf.float64)
            x0_mean = tf.squeeze(x0_mean)

            x0_cov = tf.convert_to_tensor(x0_cov(th), tf.float64)
            x0_cov.set_shape([n, n])

            x0_dist = MultivariateNormalFullCovariance(x0_mean, x0_cov,
                                                       name='x0_dist')

            Q = tf.convert_to_tensor(self.__tf_w_cov(th), tf.float64)
            Q.set_shape([p, p])

            w_mean = self.__w_mean.flatten()
            w_dist = MultivariateNormalFullCovariance(w_mean, Q, name='w_dist')

            R = tf.convert_to_tensor(self.__tf_v_cov(th), tf.float64)
            R.set_shape([m, m])
            v_mean = self.__v_mean.flatten()
            v_dist = MultivariateNormalFullCovariance(v_mean, R, name='v_dist')

            def sim_obs(x):
                v = v_dist.sample()
                v = tf.reshape(v, [m, 1])
                y = H @ x + v  # the syntax is valid for Python >= 3.5
                return y

            def sim_loop_cond(x, y, t, k):
                N = tf.stack([tf.shape(t)[0]])
                N = tf.reshape(N, ())
                return tf.less(k, N-1)

            def sim_loop_body(x, y, t, k):

                # TODO: this should be function of time
                u_t_k = tf.slice(u, [0, k], [r, 1])

                def state_propagate(x):
                    w = w_dist.sample()
                    w = tf.reshape(w, [p, 1])
                    Fx = tf.matmul(F, x, name='Fx')
                    Cu = tf.matmul(C, u_t_k, name='Cu')
                    Gw = tf.matmul(G, w, name='Gw')
                    x = Fx + Cu + Gw
                    return x

                tk = tf.slice(t, [k], [2], 'tk')

                x_k = x[:, -1]
                x_k = tf.reshape(x_k, [n, 1])

                x_k = state_propagate(x_k)

                y_k = sim_obs(x_k)

                # TODO: stack instead of concat
                x = tf.concat([x, x_k], 1)
                y = tf.concat([y, y_k], 1)

                k = k + 1

                return x, y, t, k

            x = x0_dist.sample(name='x0_sample')
            x = tf.reshape(x, [n, 1], name='x')

            # this zeroth measurement should be thrown away
            y = sim_obs(x)
            k = tf.constant(0, name='k')

            shape_invariants = [tf.TensorShape([n, None]),
                                tf.TensorShape([m, None]),
                                t.get_shape(),
                                k.get_shape()]

            sim_loop = tf.while_loop(sim_loop_cond, sim_loop_body,
                                     [x, y, t, k], shape_invariants,
                                     name='sim_loop')

            self.__sim_loop_op = sim_loop

    # defines graph
    def __define_likelihood_computation(self):

        self.__lik_graph = tf.Graph()
        lik_graph = self.__lik_graph

        r = self.__r
        m = self.__m
        n = self.__n
        p = self.__p

        x0_mean = self.__tf_x0_mean
        x0_cov = self.__tf_x0_cov

        with lik_graph.as_default():
            # FIXME: Don't Repeat Yourself (in simulation and here)
            th = tf.placeholder(tf.float64, shape=[None], name='th')
            u = tf.placeholder(tf.float64, shape=[r, None], name='u')
            t = tf.placeholder(tf.float64, shape=[None], name='t')
            y = tf.placeholder(tf.float64, shape=[m, None], name='y')

            N = tf.stack([tf.shape(t)[0]])
            N = tf.reshape(N, ())

            F = tf.convert_to_tensor(self.__tf_F(th), tf.float64)
            F.set_shape([n, n])

            C = tf.convert_to_tensor(self.__tf_C(th), tf.float64)
            C.set_shape([n, r])

            G = tf.convert_to_tensor(self.__tf_G(th), tf.float64)
            G.set_shape([n, p])

            H = tf.convert_to_tensor(self.__tf_H(th), tf.float64)
            H.set_shape([m, n])

            x0_mean = tf.convert_to_tensor(x0_mean(th), tf.float64)
            x0_mean.set_shape([n, 1])

            P_0 = tf.convert_to_tensor(x0_cov(th), tf.float64)
            P_0.set_shape([n, n])

            Q = tf.convert_to_tensor(self.__tf_w_cov(th), tf.float64)
            Q.set_shape([p, p])

            R = tf.convert_to_tensor(self.__tf_v_cov(th), tf.float64)
            R.set_shape([m, m])

            I = tf.eye(n, n, dtype=tf.float64)

            def lik_loop_cond(k, P, S, t, u, x, y, yhat):
                return tf.less(k, N-1)

            def lik_loop_body(k, P, S, t, u, x, y, yhat):

                # TODO: this should be function of time
                u_t_k = tf.slice(u, [0, k], [r, 1])

                # k+1, cause zeroth measurement should not be taken into account
                y_k = tf.slice(y, [0, k+1], [m, 1])

                t_k = tf.slice(t, [k], [2], 't_k')

                # TODO: extract Kalman filter to a separate class
                def state_predict(x):
                    Fx = tf.matmul(F, x, name='Fx')
                    Cu = tf.matmul(C, u_t_k, name='Cu')
                    x = Fx + Cu
                    return x

                def covariance_predict(P):
                    GQtG = tf.matmul(G @ Q, G, transpose_b=True)
                    PtF = tf.matmul(P, F, transpose_b=True)
                    P = tf.matmul(F, P) + PtF + GQtG
                    return P

                x = state_predict(x)

                P = covariance_predict(P)

                yh = H @ x

                yhat = tf.concat([yhat, yh], axis=1)

                E = y_k - yh

                B = tf.matmul(H @ P, H, transpose_b=True) + R
                invB = tf.matrix_inverse(B)

                K = tf.matmul(P, H, transpose_b=True) @ invB

                S_k = tf.matmul(E, invB @ E, transpose_a=True)
                S_k = 0.5 * (S_k + tf.log(tf.matrix_determinant(B)))

                S = S + S_k

                # state update
                x = x + tf.matmul(K, E)

                # covariance update
                P = (I - K @ H) @ P

                k = k + 1

                return k, P, S, t, u, x, y, yhat

            k = tf.constant(0, name='k')
            P = P_0
            S = tf.constant(0.0, dtype=tf.float64, shape=[1, 1], name='S')
            x = x0_mean
            yhat = H @ x

            shape_invariants = [k.get_shape(), P.get_shape(), S.get_shape(),
                                t.get_shape(), u.get_shape(), x.get_shape(),
                                y.get_shape(), tf.TensorShape([m, None])]

            # TODO: make a named tuple of named list
            lik_loop = tf.while_loop(lik_loop_cond, lik_loop_body,
                                     [k, P, S, t, u, x, y, yhat],
                                     shape_invariants,
                                     name='lik_loop')

            dS = tf.gradients(lik_loop[2], th)

            self.__lik_loop_op = lik_loop
            self.__dS = dS

    def __isObservable(self, th=None):
        if th is None:
            th = self.__th
        F = np.array(self.__F(th))
        H = np.array(self.__H(th))
        n = self.__n
        obsv_matrix = control.obsv(F, H)
        rank = np.linalg.matrix_rank(obsv_matrix)
        return rank == n

    def __isControllable(self, th=None):
        if th is None:
            th = self.__th
        F = np.array(self.__F(th))
        C = np.array(self.__C(th))
        n = self.__n
        ctrb_matrix = control.ctrb(F, C)
        rank = np.linalg.matrix_rank(ctrb_matrix)
        return rank == n

    # FIXME: fix to discrete
    def __isStable(self, th=None):
        if th is None:
            th = self.__th
        F = np.array(self.__F(th))
        eigv = np.linalg.eigvals(F)
        abs_vals = np.abs(eigv)
        return np.all(abs_vals < 1)

    def __validate(self, th=None):
        # FIXME: do not raise exceptions
        # TODO: prove, print matrices and their criteria
        if not self.__isControllable(th):
            # raise Exception('''Model is not controllable. Set different
            #                structure or parameters values''')
            pass

        if not self.__isStable(th):
            # raise Exception('''Model is not stable. Set different structure or
            #                parameters values''')
            pass

        if not self.__isObservable(th):
            # raise Exception('''Model is not observable. Set different
            #                structure or parameters values''')
            pass

    def sim(self, u, th=None):
        if th is None:
            th = self.__th

        r = self.__r
        u = np.array(u).reshape([r, -1])
        k = u.shape[1]
        t = np.linspace(0, k-1, k)

        self.__validate(th)
        g = self.__sim_graph

        if t.shape[0] != u.shape[1]:
            raise Exception('''t.shape[0] != u.shape[1]''')

        # run simulation graph
        with tf.Session(graph=g) as sess:
            t_ph = g.get_tensor_by_name('t:0')
            th_ph = g.get_tensor_by_name('th:0')
            u_ph = g.get_tensor_by_name('u:0')
            rez = sess.run(self.__sim_loop_op, {th_ph: th, t_ph: t, u_ph: u})

        return rez[1]

    def yhat(self, u, y, th=None):
        if th is None:
            th = self.__th

        k = u.shape[1]
        t = np.linspace(0, k-1, k)

        # to numpy 1D array
        th = np.array(th).squeeze()

        # self.__validate(th)
        g = self.__lik_graph

        if t.shape[0] != u.shape[1]:
            raise Exception('''t.shape[0] != u.shape[1]''')

        # run lik graph
        with tf.Session(graph=g) as sess:
            t_ph = g.get_tensor_by_name('t:0')
            th_ph = g.get_tensor_by_name('th:0')
            u_ph = g.get_tensor_by_name('u:0')
            y_ph = g.get_tensor_by_name('y:0')
            rez = sess.run(self.__lik_loop_op, {th_ph: th, t_ph: t, u_ph: u,
                                                y_ph: y})

        # TODO: make rez namedtuple
        yhat = rez[-1]
        return yhat

    def lik(self, u, y, th=None):

        # hack continuous to discrete system
        k = u.shape[1]
        t = np.linspace(0, k-1, k)

        if th is None:
            th = self.__th

        # to numpy 1D array
        th = np.array(th).squeeze()

        # self.__validate(th)
        g = self.__lik_graph

        # TODO: check for y also
        if t.shape[0] != u.shape[1]:
            raise Exception('''t.shape[0] != u.shape[1]''')

        # run lik graph
        with tf.Session(graph=g) as sess:
            t_ph = g.get_tensor_by_name('t:0')
            th_ph = g.get_tensor_by_name('th:0')
            u_ph = g.get_tensor_by_name('u:0')
            y_ph = g.get_tensor_by_name('y:0')
            rez = sess.run(self.__lik_loop_op, {th_ph: th, t_ph: t, u_ph: u,
                                                y_ph: y})

        # hack to discrete
        N = len(t)
        m = y.shape[0]
        S = rez[2]
        S = S + N*m * 0.5 + np.log(2*math.pi)

        return np.squeeze(S)

    def __L(self, th, u, y):
        return self.lik(u, y, th)

    def __dL(self, th, u, y):
        return self.dL(u, y, th)

    def dL(self, u, y, th=None):
        if th is None:
            th = self.__th

        # hack continuous to discrete system
        k = u.shape[1]
        t = np.linspace(0, k-1, k)

        # to 1D numpy array
        th = np.array(th).squeeze()

        # self.__validate(th)
        g = self.__lik_graph

        if t.shape[0] != u.shape[1]:
            raise Exception('''t.shape[0] != u.shape[1]''')

        # run lik graph
        with tf.Session(graph=g) as sess:
            t_ph = g.get_tensor_by_name('t:0')
            th_ph = g.get_tensor_by_name('th:0')
            u_ph = g.get_tensor_by_name('u:0')
            y_ph = g.get_tensor_by_name('y:0')
            rez = sess.run(self.__dS, {th_ph: th, t_ph: t, u_ph: u, y_ph: y})

        return rez[0]

    # TODO: return results as a named tuple or dictionary
    # TODO: bounds
    def mle_fit(self, th, u, y, bounds=None):
        # TODO: call slsqp, check u.shape
        th0 = th
        rez = scipy.optimize.minimize(self.__L, th0, args=(u, y),
                                      bounds=bounds, jac=self.__dL)
        return rez


    def round_weights(self, p, v):
        return np.array(self.round_plan([0, p], v)[1])

    def round_plan(self, plan, v):
        plan = copy.deepcopy(plan)
        p = np.array(plan[1])
        q = len(p)
        sigmaI = np.ceil((v - q) * p)  # 1
        sigmaII = np.floor(v * p)
        vI = v - np.sum(sigmaI)  # 2
        vII = v - np.sum(sigmaII)
        if vI < vII:
            sigma = sigmaI
            v1 = int(vI)
        else:
            sigma = sigmaII
            v1 = int(vII)

        s = np.zeros(q)

        vps = v * p - sigma
        vps_id = [i for i in range(len(vps))]
        vps_t = [(val, key) for val, key in zip(vps, vps_id)]

        vps_t = sorted(vps_t, key=operator.itemgetter(0))
        sorted_id = [elem[1] for elem in vps_t]

        for j in range(q):
            if vps_id[j] in sorted_id[:v1]:
                s[j] = 1
            else:
                s[j] = 0

        p = (sigma + s) / v
        plan[1] = p.tolist()
        return plan

    def mle_fit_plan(self, plan, v, th=None, bounds=None):
        th0 = th
        X, p = plan

    def fim(self, u, x0=None, th=None):
        """
        'u' is 2d numpy array [r x N]
        """
        if th is None:
            th = self.__th
        else:
            th = np.array(th)

        s = len(th)
        n = self.__n

        lst = list()
        lst.append(self.__F)
        lst.append(self.__C)
        lst.append(self.__G)
        lst.append(self.__H)
        lst.append(self.__w_cov)
        lst.append(self.__v_cov)
        lst.append(self.__x0_mean)
        lst.append(self.__x0_cov)

        # eval
        jlst = [autograd.jacobian(f)(th) for f in lst]

        # TODO: refactor?
        jlst = [[np.squeeze(j, 2) for j in np.dsplit(jel, s)] for jel in jlst]

        dF, dC, dG, dH, dQ, dR, dX0, dP0 = jlst

        dX0 = [dX0_i.reshape([n, 1]) for dX0_i in dX0]

        # eval
        F, C, G, H, Q, R, X0, P0 = [f(th) for f in lst]

        X0 = X0.reshape([n, 1])

        if x0 is not None:
            # on reshape fail exception will be raised
            X0 = np.array(x0).reshape([n, 1])

        C_A = np.vstack(dC)
        C_A = np.vstack([C, C_A])

        M = np.zeros([s, s])

        t = np.transpose
        inv = np.linalg.inv
        Sp = np.trace
        Pe = P0
        dPe = dP0
        Inn = np.eye(n)
        On1 = np.zeros([n, 1])

        # TODO: cast every thing to np.matrix and use '*' multiplication syntax

        def F_A_f(F, dF, H, K_):
            _1st_col = [dF_i - K_ @ dH_i for dF_i, dH_i in zip(dF, dH)]
            _1st_col = np.vstack(_1st_col)

            bdiag = scipy.linalg.block_diag(*[F - K_ @ H] * s)
            rez = np.hstack([_1st_col, bdiag])

            _1st_row = np.hstack([np.zeros([n, n])] * s)
            _1st_row = np.hstack([F, _1st_row])

            rez = np.vstack([_1st_row, rez])
            return rez

        def X_Ap_f(F_A, X_Ap, u, k):
            if k == 0:
                F_ = np.vstack(dF)
                F_ = np.vstack([F, F_])

                # force dX0_i to be 2D array
                FdX0 = [F @ np.array(dX0_i, ndmin=2) for dX0_i in dX0]
                FdX0 = np.vstack(FdX0)
                OFdX0 = np.vstack([On1, FdX0])

                # u[:,[k]] - get k-th column as column vector
                return F_ @ X0 + OFdX0 + C_A @ u[:, [0]]
            elif k > 0:
                return F_A @ X_Ap + C_A @ u[:, [k]]

        def Cf(i):
            i = i + 1
            zeros = [np.zeros([n, n])] * i
            zeros = np.hstack(zeros) if i else []
            C = np.hstack([zeros, np.eye(n)]) if i else np.eye(n)
            zeros = [np.zeros([n, n])] * (s-i)
            zeros = np.hstack(zeros) if s-i else []
            C = np.hstack([C, zeros]) if s-i else C
            return C

        u = np.array(u, ndmin=2)
        N = u.shape[1]

        if u.shape[0] != C.shape[1]:
            raise Exception('invalid shape of \'u\'')

        for k in range(N):
            if k == 0:
                E_A = np.zeros([n*(s+1), n*(s+1)])
                X_Ap = X_Ap_f(None, None, u, k)
                F_A = None
                K_A = None
                B = None
            elif k > 0:
                E_A = F_A @ E_A @ t(F_A) + K_A @ B @ t(K_A)
                X_Ap = X_Ap_f(F_A, X_Ap, u, k)

            # Pp, B, K, Pu, K_
            Pp = F @ Pe @ t(F) + G @ Q @ t(G)
            B = H @ Pp @ t(H) + R
            invB = inv(B)
            K = Pp @ t(H) @ invB
            Pu = (Inn - K @ H) @ Pp
            K_ = F @ K

            F_A = F_A_f(F, dF, H, K_)

            # TODO: numba jit it
            dPp = [dF_i @ Pe @ t(F) + F @ dPe_i @ t(F) + F @ Pe @ t(dF_i) +
                   dG_i @ Q @ t(G) + G @ dQ_i @ t(G) + G @ Q @ t(dG_i)
                   for dF_i, dPe_i, dG_i, dQ_i in zip(dF, dPe, dG, dQ)]

            dB = [dH_i @ Pp @ t(H) + H @ dPp_i @ t(H) + H @ Pp @ t(dH_i) + dR_i
                  for dH_i, dPp_i, dR_i in zip(dH, dPp, dR)]

            dK = [(dPp_i @ t(H) + Pp @ t(dH_i) - Pp @ t(H) @ invB @ dB_i) @ invB
                  for dPp_i, dH_i, dB_i in zip(dPp, dH, dB)]

            dPu = [(Inn - K @ H) @ dPp_i - (dK_i @ H + K @ dH_i) @ Pp
                   for dPp_i, dK_i, dH_i in zip(dPp, dK, dH)]

            dK_ = [dF_i @ K + F @ dK_i for dF_i, dK_i in zip(dF, dK)]

            K_A = np.vstack(dK_)
            K_A = np.vstack([K_, K_A])

            # 8: AM
            AM = list()

            EXX = E_A + X_Ap @ t(X_Ap)

            C0 = Cf(0)

            for i, j in itertools.product(range(s), range(s)):
                S1 = Sp(C0 @ EXX @ t(C0) @ t(dH[j]) @ invB @ dH[i])
                S2 = Sp(C0 @ EXX @ t(Cf(j)) @ t(H) @ invB @ dH[i])
                S3 = Sp(Cf(i) @ EXX @ t(C0) @ t(dH[j]) @ invB @ H)
                S4 = Sp(Cf(i) @ EXX @ t(Cf(j)) @ t(H) @ invB @ H)
                S5 = 0.5 * Sp(dB[i] @ invB @ dB[j] @ invB)
                AM.append(S1 + S2 + S3 + S4 + S5)

            AM = np.array(AM).reshape([s, s])
            M = M + AM

            # update P, dP etc.
            Pe = Pu
            dPe = dPu

        return M

    def norm_fim(self, plan, th=None):
        ''' plan: list of 'x' and 'p' '''
        ''' x: 3d np array [q x r x N] '''
        ''' p: list or 1d np array '''
        x, p = plan
        x = np.array(x, ndmin=2)
        p = np.array(p)

        # FIXME, TODO: validate plan
        # for x_i, p_i in zip(x, p):
        #    if len(x_i) != self.__n:
        #        raise Exception('invalid plan: len(x_i) != n')

        Mn = 0

        # extract p, x and compute fim for every point of the plan
        for x_i, p_i in zip(x, p):
            # TODO: compute in parallel
            Mn += p_i * self.fim(u=x_i, x0=None, th=th)
        return Mn

    def d_opt_crit(self, plan, th=None):
        ''' plan: list of 'x' and 'p' '''
        ''' x: 3d np array [q x r x N] '''
        ''' p: list or 1d np array '''
        Mn = self.norm_fim(plan, th)
        sign, logdet = np.linalg.slogdet(Mn)
        return -logdet

    def __d_crit_to_opt_grad(self, plan, q, th=None):
        ''' plan is 1D np.array or list '''
        plan = np.array(plan)
        grad = self.__d_crit_to_opt_grad_f(plan, q, th)
        return grad

    # this wraps self.d_opt_crit() above
    # for scipy.optimize.minimize
    def __d_crit_to_optimize(self, plan, q, th=None):
        # unflatten plan
        r = self.__r
        p = plan[-q:]
        x = plan[:-q]
        x = np.reshape(x, [q, r, -1])
        plan = [x, p]
        crit = self.d_opt_crit(plan, th)
        return crit

    # TODO: take bounds
    def direct(self, plan0, th=None):
        ''' plan0: list of: list (or 3D np.array) of 'u' and list of 'p' '''
        n = self.__n
        r = self.__r

        x, p = plan0
        x = np.array(x)
        p = np.array(p)

        q = len(p)
        N = x.shape[-1]

        x_bounds = [(-1, 1)] * q * r * N
        p_bounds = [(0, 1)] * q
        bounds = x_bounds + p_bounds  # concat lists

        def heq(x):
            p = x[-q:]
            return np.sum(p) - 1

        constraints = {'type': 'eq', 'fun': heq}

        x0 = x.flatten()
        x0 = np.hstack([x0, p])

        rez = scipy.optimize.minimize(fun=self.__d_crit_to_optimize, x0=x0,
                                      jac=self.__d_crit_to_opt_grad,
                                      args=(q, th), method='SLSQP',
                                      constraints=constraints, bounds=bounds)
        new_plan = rez['x']

        pn = new_plan[-q:]
        xn = new_plan[:-q].reshape([q, r, N])
        new_plan = [xn, pn]

        # TODO: return loss and its jacobian values
        # return dictionary
        return new_plan, rez['fun']

    def clean(self, plan, dn=0.5, dp=0.1):
        ''' plan = [x, p], x is 3D array, p is list or 1D np array '''
        x, p = plan
        p = list(p)

        # clean by weight
        while True:
            indices = [i for i in range(len(p)) if p[i] < dp]
            if len(indices) == 0:
                break

            i = indices[0]
            x = np.delete(x, i, 0)
            p_i = p.pop(i)
            p = [p_j + p_i / len(p) for p_j in p]

        q, r, N = x.shape
        x = x.reshape([q, -1])

        # clean by distance
        while True:
            tree = scipy.spatial.cKDTree(x)
            bt = tree.query_ball_tree(tree, dn)
            lengths = [len(bt_i) for bt_i in bt]
            max_length = max(lengths)
            if max_length == 1:  # nothing to clean
                break
            else:  # clean
                i = lengths.index(max_length)
                indices = bt[i]  # get close points indices

                # merge points to new one
                new_point = [p[i] * x[i] for i in indices]
                px = sum([p[i] for i in indices])
                new_point = sum(new_point) / px

                x = np.delete(x, indices, 0)   # delete merged points
                x = np.vstack([x, new_point])  # append new point

                # merge corresponding weights to new one
                pn = sum([p[i] for i in range(len(p)) if i in indices])

                # delete old weights
                p = [p[i] for i in range(len(p)) if i not in indices]

                p.append(pn)  # add new weight

        q = len(p)
        x = x.reshape([q, r, N])

        return x, p

    # wraps fim()
    def __mu(self, u, M_plan, th):
        M = self.fim(u=u, th=th)
        return -np.trace(np.linalg.inv(M_plan) @ M)

    def crit_tau(self, tau, a, plan, th):
        x, p = plan
        p = list(p)
        x = np.array(x)
        a = np.expand_dims(a, 0)  # TODO: set shape explicitly
        x = np.concatenate([x, a])
        p = [p_i * (1 - tau) for p_i in p]
        p.append(tau)
        plan = [x, p]
        crit = self.d_opt_crit(plan, th)
        return crit

    def rand_plan(self, N, bounds=None):
        r = self.__r
        s = len(self.__th)
        q = int((s + 1) * s / 2 + 1)
        x = np.random.uniform(-1, 1, [q, r, N])
        p = [1 / q] * q
        plan = [x, p]
        crit = self.d_opt_crit(plan)
        return x, p, crit

    def dual(self, plan, th=None, d=0.05):
        ''' plan '''
        dmu = autograd.grad(self.__mu)  # this is *not* time consuming

        plan = copy.deepcopy(plan)

        if th is None:
            th = self.__th
        else:
            th = np.array(th)

        eta = len(th)
        n = self.__n
        r = self.__r
        X, p = plan  # TODO: make plan class
        N = X.shape[-1]

        crit_tau_grad = autograd.grad(self.crit_tau)

        x_bounds = [(-1, 1)] * r * N

        while True:
            M_plan = self.norm_fim(plan, th)

            while True:
                x_guess = np.random.uniform(-1, 1, [r, N])  # FIXME

                # nlopts <- list(xtol_rel=1e-3, maxeval=1e3)
                rez = scipy.optimize.minimize(fun=self.__mu, x0=x_guess,
                                              args=(M_plan, th),
                                              method='SLSQP', jac=dmu,
                                              bounds=x_bounds, tol=None,
                                              options=None)
                x_opt = rez['x'].reshape([r, N])
                mu = -rez['fun']

                if abs(mu - eta) <= d:
                    return plan, self.d_opt_crit(plan)

                if mu > eta:
                    break

            while True:
                # XXX: this was needed to get non singular tau value,
                # not sure if it is still needed
                tau_guess = np.random.uniform(size=1)
                tau_crit = self.crit_tau(tau_guess, x_opt,
                                         copy.deepcopy(plan), th)
                if not np.isnan(tau_crit):
                    break

            rez = scipy.optimize.minimize(fun=self.crit_tau, x0=tau_guess,
                                          args=(x_opt, copy.deepcopy(plan), th),
                                          bounds=[(0, 1)],
                                          method='SLSQP', jac=crit_tau_grad)

            tau_opt = rez['x']

            # add x_opt, tau_opt to plan
            X, p = plan
            x_opt = np.expand_dims(x_opt, 0)
            X = np.concatenate([X, x_opt]) # FIXME
            tau_opt = tau_opt[0]
            p = [p_i - tau_opt / len(p) for p_i in p]
            p.append(tau_opt)
            plan = [X, p]

            # clean plan
            plan = self.clean(copy.deepcopy(plan))

            # continue
