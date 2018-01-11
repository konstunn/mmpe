
import os
import math
import tensorflow as tf
import control
import autograd.numpy as np
import autograd
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
import scipy
import copy
import operator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Plan(object):
    def __init__(self, q, r, N):
        pass

    def clean(self):
        pass

    def add(self, x, p):
        pass

    def rand(self):
        pass

    def round(self, v):
        pass


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
        self.__s = len(th)

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
        self.__define_fim_computation()

        self.__d_crit_to_opt_grad_f = autograd.grad(self.__d_crit_to_optimize)

    # FIXME: fix for continuous
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

                # FIXME: fix for continuous
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

    # derivative of matrix 'a' w.r.t vector 'x'
    def __matderiv(self, a, x):

        def comp_grads(tensor, var_list):
            grads = tf.gradients(tensor, var_list)
            return [grad if grad is not None else tf.zeros_like(var)
                    for var, grad in zip(var_list, grads)]

        s = x.get_shape().as_list()[0]  # XXX: what if x shape is unknown
        N = tf.size(a)
        m, n = a.get_shape()

        a = tf.reshape(a, [N])

        def cond(i, N, rez):
            return tf.less(i, N)

        def body(i, N, rez):
            elem = tf.slice(a, [i], [1])
            elem = comp_grads(elem, [x])
            elem = tf.reshape(elem, [s, 1])
            rez = tf.concat([rez, elem], 1)
            i = i + 1
            return i, N, rez

        shape_invariants = [tf.TensorShape([]),
                            tf.TensorShape([]),
                            tf.TensorShape([s, None])]

        elem = tf.slice(a, [0], [1])
        rez = comp_grads(elem, [x])[0]
        rez = tf.reshape(rez, [s, 1])

        loop = tf.while_loop(cond, body, [1, N, rez], shape_invariants)

        rez = loop[2]

        rez = tf.reshape(rez, [s, m, n])
        return rez

    def __define_fim_computation(self):

        fim_graph = self.__fim_graph = tf.Graph()

        r = self.__r
        m = self.__m
        n = self.__n
        p = self.__p
        s = self.__s

        x0_mean = self.__tf_x0_mean
        x0_cov = self.__tf_x0_cov

        with fim_graph.as_default():
            th = tf.placeholder(tf.float32, shape=[s], name='th')
            u = tf.placeholder(tf.float32, shape=[r, None], name='u')
            t = tf.placeholder(tf.float32, shape=[None], name='t')

            N = tf.stack([tf.shape(t)[0]])
            N = tf.reshape(N, ())
            u = tf.reshape(u, [N, r, 1])

            F = tf.convert_to_tensor(self.__tf_F(th), tf.float32)
            F.set_shape([n, n])

            C = tf.convert_to_tensor(self.__tf_C(th), tf.float32)
            C.set_shape([n, r])

            G = tf.convert_to_tensor(self.__tf_G(th), tf.float32)
            G.set_shape([n, p])

            H = tf.convert_to_tensor(self.__tf_H(th), tf.float32)
            H.set_shape([m, n])

            x0_mean = tf.convert_to_tensor(x0_mean(th), tf.float32)
            x0_mean.set_shape([n, 1])

            P_0 = tf.convert_to_tensor(x0_cov(th), tf.float32)
            P_0.set_shape([n, n])

            Q = tf.convert_to_tensor(self.__tf_w_cov(th), tf.float32)
            Q.set_shape([p, p])

            R = tf.convert_to_tensor(self.__tf_v_cov(th), tf.float32)
            R.set_shape([m, m])

            dF = self.__matderiv(F, th)
            dC = self.__matderiv(C, th)
            dG = self.__matderiv(G, th)
            dH = self.__matderiv(H, th)
            dQ = self.__matderiv(Q, th)
            dR = self.__matderiv(R, th)
            dP = self.__matderiv(P_0, th)
            dX_0 = self.__matderiv(x0_mean, th)

            # compute exponential decay
            def mat_exp(F, t):
                n = F.get_shape().as_list()[0]
                def ode(T, t):
                    return -T @ F
                T0 = tf.eye(n)
                T = tf.contrib.integrate.odeint(ode, T0, t)
                return tf.reverse(T, [0])

            # TODO: try AD
            # compute exponential decay derivative w.r.t time
            def mat_exp_deriv(F, dF, T, t_grid):
                n = F.get_shape().as_list()[0]
                s = dF.get_shape().as_list()[0]

                def ode(dT, t):
                    dT_F = tf.map_fn(lambda dTi: dTi @ F, dT)
                    rhos = tf.abs(t_grid - t)
                    min_rho = tf.reduce_min(rhos)
                    idx = tf.where(tf.equal(rhos, min_rho))[0][0]
                    T_t = tf.slice(T, [idx, 0, 0], [1, n, n])[0]
                    T_dF = tf.map_fn(lambda dFi: T_t @ dFi, dF)
                    return -dT_F - T_dF
                dT0 = tf.zeros([s, n, n])
                dT = tf.contrib.integrate.odeint(ode, dT0, t_grid)
                return tf.reverse(dT, [0])

            # if k == 0
            def comp_x_A_0(T, dT, C, dC, x_0, dx_0, u_k, t_grid):
                # XXX: 'u_k' is 2-rank tensor with shapes [r, 1]
                n = C.get_shape().as_list()[0]
                s = dC.get_shape().as_list()[0]

                T_0 = T[0]
                dT_0 = dT[0]

                T_x_0 = T_0 @ x_0
                dT_x_0 = tf.map_fn(lambda dTi: dTi @ x_0, dT_0)
                T_dx_0 = tf.map_fn(lambda dx_0_i: T_0 @ dx_0_i, dx_0)

                def integral(x_a, t):
                    rhos = tf.abs(t_grid - t)
                    min_rho = tf.reduce_min(rhos)
                    idx = tf.where(tf.equal(rhos, min_rho))[0][0]
                    T_t = T[idx]
                    dT_t = dT[idx]
                    T_C_u = T_t @ C @ u_k
                    dT_C_u = tf.map_fn(lambda dTi: dTi @ C @ u_k, dT_t)
                    T_dC_u = tf.map_fn(lambda dCi: T_t @ dCi @ u_k, dC)
                    derivs = dT_C_u + T_dC_u
                    derivs = tf.concat(tf.unstack(derivs), axis=0)
                    return tf.concat([T_C_u, derivs], axis=0)

                init_val = tf.zeros([n*(s+1), 1])

                int_vals = tf.contrib.integrate.odeint(integral, init_val,
                                                       t_grid)

                # reverse to slice the last (first)
                int_vals = tf.reverse(int_vals, [0])
                int_val = tf.slice(int_vals, [0, 0, 0], [1, n*(s+1), 1])[0]

                dT_x_0__T_dx_0 = dT_x_0 + T_dx_0
                dT_x_0__T_dx_0 = tf.concat(tf.unstack(dT_x_0__T_dx_0), axis=0)
                return tf.concat([T_x_0, dT_x_0__T_dx_0], axis=0) + int_val

            def block_diag_matrix(block, N):
                n = block.get_shape().as_list()[0]

                def cond(k, rez):
                    return tf.less(k, N)

                def body(k, rez):
                    n_rows = tf.pad(block, [[0, 0], [k*n, (N-1-k)*n]])
                    rez = tf.concat([rez, n_rows], axis=0)
                    return k+1, rez

                shape_invariants = [tf.TensorShape([]),
                                    tf.TensorShape([None, None])]

                _1st_n_rows = tf.pad(block, [[0, 0], [0, (N-1)*n]])
                return tf.while_loop(cond, body, [1, _1st_n_rows],
                                     shape_invariants)[1]

            def comp_T_A(H, dH, T, dT, K):
                n = T.get_shape().as_list()[0]
                s = dT.get_shape().as_list()[0]
                K_ = T @ K
                phi_a = T - K_ @ H
                block_diag = block_diag_matrix(phi_a, s)
                first_row = tf.pad(T, [[0, 0], [0, n*s]])
                K_dH = tf.map_fn(lambda dHi: K_ @ dHi, dH)
                derivs = tf.unstack(dT - K_dH)
                derivs = tf.concat(derivs, axis=0)
                rez = tf.concat([derivs, block_diag], axis=1)
                rez = tf.concat([first_row, rez], axis=0)
                return rez

            def predict_P_dP(F, dF, P, dP, G, dG, Q, dQ, t_grid):

                def build_F_dF_block_diag(F, dF):
                    n = F.get_shape().as_list()[0]
                    s = dF.get_shape().as_list()[0]
                    F_block = block_diag_matrix(F, s)
                    first_row = tf.pad(F, [[0, 0], [0, n*s]])
                    derivs = tf.unstack(dF)
                    derivs = tf.concat(derivs, axis=0)
                    rez = tf.concat([derivs, F_block], axis=1)
                    rez = tf.concat([first_row, rez], axis=0)
                    return rez

                def build_P_dP_block_diag(PdP):
                    n = PdP.get_shape().as_list()[1]
                    s = int(PdP.get_shape().as_list()[0] / n - 1)
                    P = tf.slice(PdP, [0, 0], [n, n])
                    dP = tf.slice(PdP, [n, 0], [n*s, n])
                    P_block = block_diag_matrix(P, s)
                    first_row = tf.pad(P, [[0, 0], [0, n*s]])
                    rez = tf.concat([dP, P_block], axis=1)
                    rez = tf.concat([first_row, rez], axis=0)
                    return rez

                def build_FdF_transposed(F, dF):
                    F = tf.stack([F])
                    FdF = tf.concat([F, dF], axis=0)
                    FdF_t = tf.map_fn(lambda FdF_i: tf.transpose(FdF_i), FdF)
                    FdF_t = tf.unstack(FdF_t)
                    FdF_t = tf.concat(FdF_t, axis=0)
                    return FdF_t

                def comp_GdG_Q_Gt(G, dG, Q):
                    G_t = tf.transpose(G)
                    GQGt = G @ Q @ G_t
                    rest = tf.map_fn(lambda dG_i: dG_i @ Q @ G_t, dG)
                    rest = tf.unstack(rest)
                    rest = tf.concat(rest, 0)
                    rez = tf.concat([GQGt, rest], 0)
                    return rez

                def comp_GdQGt(G, dQ):
                    n = G.get_shape().as_list()[0]
                    G_t = tf.transpose(G)
                    GdQGt = tf.map_fn(lambda dQ_i: G @ dQ_i @ G_t, dQ)
                    GdQGt = tf.unstack(GdQGt)
                    GdQGt = tf.concat(GdQGt, axis=0)
                    return tf.pad(GdQGt, [[n, 0], [0, 0]])

                def comp_GQdG_t(G, dG, Q):
                    n = G.get_shape().as_list()[0]
                    dGt = tf.map_fn(lambda dG_i: tf.transpose(dG_i), dG)
                    GQdG_t = tf.map_fn(lambda dGt_i: G @ Q @ dGt_i, dGt)
                    GQdG_t = tf.concat(tf.unstack(GQdG_t), axis=0)
                    return tf.pad(GQdG_t, [[n, 0], [0, 0]])

                def ode(PdP, t):
                    F_dF_bdiag = build_F_dF_block_diag(F, dF)
                    P_dP_bdiag = build_P_dP_block_diag(PdP)
                    FdF_t = build_FdF_transposed(F, dF)
                    dGQGt = comp_GdG_Q_Gt(G, dG, Q)
                    GdQGt = comp_GdQGt(G, dQ)
                    GQdGt = comp_GQdG_t(G, dG, Q)
                    PdP = F_dF_bdiag @ PdP + P_dP_bdiag @ FdF_t + dGQGt + GdQGt
                    PdP = PdP + GQdGt  # TODO: may tf.stack and tf.reduce_sum
                    return PdP

                n = F.get_shape().as_list()[0]
                s = dF.get_shape().as_list()[0]

                dP = tf.unstack(dP)
                dP = tf.concat(dP, axis=0)
                PdP = tf.concat([P, dP], axis=0)
                PdP = tf.contrib.integrate.odeint(ode, PdP, t_grid, rtol=1e-2,
                                                  atol=1e-3)
                PdP = PdP[-1]
                P = tf.slice(PdP, [0, 0], [n, n])
                dP = tf.slice(PdP, [n, 0], [n*s, n])
                dP = tf.reshape(dP, [s, n, n])
                return P, dP

            def comp_dB(H, dH, P, dP, dR):
                Ht = tf.transpose(H)
                dH_P_Ht = tf.map_fn(lambda dH_i: dH_i @ P @ Ht, dH)
                H_dP_Ht = tf.map_fn(lambda dP_i: H @ dP_i @ Ht, dP)
                H_P_dHt = tf.map_fn(lambda dH_i: H @ P @ tf.transpose(dH_i), dH)
                return dH_P_Ht + H_dP_Ht + H_P_dHt + dR

            def comp_dK(B, dB, H, dH, P, dP):
                Ht = tf.transpose(H)
                dHt = tf.map_fn(lambda dH_i: tf.transpose(dH_i), dH)
                invB = tf.matrix_inverse(B)
                dP_Ht_invB = tf.map_fn(lambda dP_i: dP_i @ Ht @ invB, dP)
                P_dHt_invB = tf.map_fn(lambda dHt_i: P @ dHt_i @ invB, dHt)
                P_Ht_invB_dB_invB = tf.map_fn(
                    lambda dB_i: P @ Ht @ invB @ dB_i @ invB, dB)
                P_Ht_invB = P @ Ht @ invB
                return dP_Ht_invB + P_dHt_invB - P_Ht_invB - P_Ht_invB_dB_invB

            def update_P(H, K, P):
                m = K.get_shape().as_list()[0]
                I = tf.eye(m)
                return (I - K @ H) @ P

            def update_dP(H, dH, K, dK, P, dP):
                m = K.get_shape().as_list()[0]
                I = tf.eye(m)
                I_K_H_dP = tf.map_fn(lambda dP_i: (I - K @ H) @ dP_i, dP)
                dK_H = tf.map_fn(lambda dK_i: dK_i @ H, dK)
                K_dH = tf.map_fn(lambda dH_i: K @ dH_i, dH)
                return I_K_H_dP - (dK_H + K_dH)

            def build_K_A(K_, dK_):
                dK_ = tf.unstack(dK_)
                dK_ = tf.concat(dK_, axis=0)
                return tf.concat([K_, dK_], axis=0)

            def comp_a_A(T, dT, C, dC, u_k, t_grid):
                # 'u_k' is 2-rank tensor with shapes [r, 1]

                a = C @ u_k
                da = tf.map_fn(lambda dC_i: dC_i @ u_k, dC)

                def comp_a_A_k(T_k, dT_k):
                    Tk_a = tf.stack([T_k @ a])
                    dTk_a = tf.map_fn(lambda dT_i: dT_i @ a, dT_k)
                    Tk_da = tf.map_fn(lambda da_i: T_k @ da_i, da)
                    derivs = dTk_a + Tk_da
                    return tf.concat([Tk_a, derivs], axis=0)  # 3-rank tensor

                # 'a_A' is 3-rank tensor
                def ode(a_A, t):
                    rhos = tf.abs(t_grid - t)
                    min_rho = tf.reduce_min(rhos)
                    idx = tf.where(tf.equal(rhos, min_rho))[0][0]
                    return comp_a_A_k(T[idx], dT[idx])

                a_A_0 = comp_a_A_k(T[0], dT[0])

                a_A = tf.contrib.integrate.odeint(ode, a_A_0, t_grid)[-1]

                return tf.concat(tf.unstack(a_A), axis=0)  # 2-rank tensor

            def S(n, s, i):
                zeros = tf.zeros([n, n*i])
                eye = tf.eye(n)
                rez = tf.concat([zeros, eye], axis=1)
                zeros = tf.zeros([n, n*(s-i)])
                rez = tf.concat([rez, zeros], axis=1)
                rez.set_shape([n, n*(s+1)])
                return rez

            def comp_dM(Sigma_A, x_A, H, dH, invB, dB):
                s = dH.get_shape().as_list()[0]
                Sigma_A__x_A__x_A_T = Sigma_A + x_A @ tf.transpose(x_A)

                def cond(dM, l):
                    return tf.less(l, s*s)

                def body(dM, l):
                    i = tf.div(l, s)
                    j = tf.mod(l, s)
                    S0 = S(n, s, 0)
                    Sj = S(n, s, j)
                    Si = S(n, s, i)
                    SjT = tf.transpose(Sj)
                    S0t = tf.transpose(S0)
                    dHjT = tf.transpose(dH[j])
                    dHi = dH[i]
                    Ht = tf.transpose(H)
                    _1 = S0 @ Sigma_A__x_A__x_A_T @ S0t @ dHjT @ invB @ dHi
                    _1 = tf.trace(_1)
                    _2 = S0 @ Sigma_A__x_A__x_A_T @ SjT @ Ht @ invB @ dHi
                    _2 = tf.trace(_2)
                    _3 = Si @ Sigma_A__x_A__x_A_T @ S0t @ dHjT @ invB @ H
                    _3 = tf.trace(_3)
                    _4 = Si @ Sigma_A__x_A__x_A_T @ SjT @ Ht @ invB @ H
                    _4 = tf.trace(_4)
                    _5 = tf.trace(dB[i] @ invB @ dB[j])
                    elem = _1 + _2 + _3 + _4 + _5
                    dM = tf.concat([dM, tf.stack([elem])], axis=0)
                    return dM, l+1

                dM0 = tf.zeros([0])
                l = tf.constant(0)

                shape_invariants = [tf.TensorShape([None]), l.get_shape()]

                dM = tf.while_loop(cond, body, [dM0, l], shape_invariants)[0]
                dM = tf.reshape(dM, [s, s])
                return dM

            def cond(M, k, B, K, dK, P, dP, Sigma_A, x_A):
                return tf.less(k, N-1)

            # TODO FIXME
            def first_iteration(P, dP):
                t_grid = tf.slice(t, [0], [2])
                t_grid = tf.linspace(t_grid[0], t_grid[1], 100)
                T = mat_exp(F, t_grid)
                dT = mat_exp_deriv(F, dF, T, t_grid)
                u_k = u[0]

                x_A = comp_x_A_0(T, dT, C, dC, x0_mean, dX_0, u_k, t_grid)
                shape = x_A.get_shape().as_list()[0]
                Sigma_A = tf.zeros([shape, shape])
                Pp, dPp = predict_P_dP(F, dF, P, dP, G, dG, Q, dQ, t_grid)
                Ht = tf.transpose(H)
                B = H @ Pp @ Ht + R
                invB = tf.matrix_inverse(B)
                dB = comp_dB(H, dH, Pp, dPp, dR)
                K = Pp @ Ht @ invB
                dK = comp_dK(B, dB, H, dH, Pp, dPp)
                Pu = update_P(H, K, Pp)
                dPu = update_dP(H, dH, K, dK, Pp, dPp)
                dM = comp_dM(Sigma_A, x_A, H, dH, invB, dB)
                return dM, 1, B, K, dK, Pu, dPu, Sigma_A, x_A

            def body(M, k, B, K, dK, P, dP, Sigma_A, x_A):
                t_grid = tf.slice(t, [k], [2])  # FIXME
                t_grid = tf.linspace(t_grid[0], t_grid[1], 100)
                T = mat_exp(F, t_grid)
                dT = mat_exp_deriv(F, dF, T, t_grid)
                u_k = u[k]

                a_A = comp_a_A(T, dT, C, dC, u_k, t_grid)
                T0 = T[0]
                dT0 = dT[0]
                K_ = T0 @ K
                dT_K = tf.map_fn(lambda dT_i: dT_i @ K, dT0)
                T_dK = tf.map_fn(lambda dK_i: T0 @ dK_i, dK)
                dK_ = dT_K + T_dK
                T_A = comp_T_A(H, dH, T0, dT0, K)
                K_A = build_K_A(K_, dK_)
                a_A = comp_a_A(T, dT, C, dC, u_k, t_grid)
                x_A = T_A @ x_A + a_A
                T_A_t = tf.transpose(T_A)
                K_A_t = tf.transpose(K_A)
                Sigma_A = T_A @ Sigma_A @ T_A_t + K_A @ B @ K_A_t
                Ht = tf.transpose(H)
                Pp, dPp = predict_P_dP(F, dF, P, dP, G, dG, Q, dQ, t_grid)
                B = H @ Pp @ Ht + R
                invB = tf.matrix_inverse(B)
                dB = comp_dB(H, dH, Pp, dPp, dR)
                K = Pp @ Ht @ invB
                dK = comp_dK(B, dB, H, dH, P, dP)
                Pu = update_P(H, K, Pp)
                dPu = update_dP(H, dH, K, dK, Pp, dPp)
                dM = comp_dM(Sigma_A, x_A, H, dH, invB, dB)
                dM = dM + M
                return dM, k+1, B, K, dK, Pu, dPu, Sigma_A, x_A

            variables = first_iteration(P_0, dP)

            fim_loop = tf.while_loop(cond, body, variables)[0]

            self.__fim_loop_op = fim_loop


    # defines graph, FIXME: for continuos
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
        u = np.array(u, ndmin=2)
        rez = scipy.optimize.minimize(self.__L, th0, args=(u, y),
                                      bounds=bounds, method='SLSQP',
                                      jac=self.__dL)
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

    def grad_lik_plan(self, th, plan, Y, v):
        U, p = plan
        dS = 0
        for i in range(len(U)):
            k_i = int(p[i] * v)
            for j in range(k_i):
                dS += self.__dL(th, U[i], Y[i][j])
        return dS

    def lik_plan(self, th, plan, Y, v):
        U, p = plan
        S = 0
        for i in range(len(U)):
            k_i = int(p[i] * v)
            for j in range(k_i):
                S += self.lik(U[i], Y[i][j], th)
        return S

    def mle_fit_plan(self, plan, v, th0, bounds=None):
        plan = self.round_plan(plan, v)
        U, p = plan
        q = U.shape[0]
        Y = list()
        for i in range(q):
            Y.append(list())
        for i in range(q):
            k_i = int(p[i] * v)
            for j in range(k_i):
                y = self.sim(U[i])
                Y[i].append(y)

        th = self.__th
        bounds = [(th_i-th_i*0.5, th_i+th_i*0.5) for th_i in th]

        rez = scipy.optimize.minimize(fun=self.lik_plan, x0=th0,
                                      args=(plan, Y, v),
                                      method='SLSQP', jac=self.grad_lik_plan,
                                      bounds=bounds)

        # calculate th rel tolerance
        th_e = rez['x']
        rtol_th = np.linalg.norm(th - th_e) / np.linalg.norm(th)
        rez['rtol_th'] = rtol_th

        # calc y rel tol
        y_rtols = list()
        for i in range(q):
            k_i = int(p[i] * v)
            for j in range(k_i):
                yh_e = self.yhat(U[i], Y[i][j], th_e)
                yh = self.yhat(U[i], Y[i][j])
                y_rtol = np.linalg.norm(yh_e - yh) / np.linalg.norm(yh)
                y_rtols.append(y_rtol)

        avg_y_rtol = sum(y_rtols) / len(y_rtols)
        rez['avg_y_rtol'] = avg_y_rtol
        return rez

    def fim(self, u, t, th=None):
        """
        'u' is 2d numpy array [r x N]
        """
        if th is None:
            th = self.__th

        th = np.array(th).squeeze()
        g = self.__fim_graph
        u = np.array(u, ndmin=2)

        if t.shape[0] != u.shape[1]:
            raise Exception('''t.shape[0] != u.shape[1]''')

        with tf.Session(graph=g) as sess:
            t_ph = g.get_tensor_by_name('t:0')
            th_ph = g.get_tensor_by_name('th:0')
            u_ph = g.get_tensor_by_name('u:0')
            rez = sess.run(self.__fim_loop_op, {th_ph: th, t_ph: t, u_ph: u})

        return rez

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
        _, logdet = np.linalg.slogdet(Mn)
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

        # TODO: return loss and its jacobian values
        # return dictionary
        return [xn, pn]

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
            p = [p_i / sum(p) for p_i in p]  # make sure sum(p) = 1

        q, r, N = x.shape
        x = x.reshape([q, -1])

        # clean by distance
        while True:
            tree = scipy.spatial.cKDTree(x)
            bt = tree.query_ball_tree(tree, dn)
            lengths = [len(bt_i) for bt_i in bt]
            max_length = max(lengths)
            if max_length == 1:
                break  # nothing to clean
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

        return [x, p]

    # wraps fim()
    def __mu(self, u, M_plan, th):
        M = self.fim(u=u, th=th)
        return -np.trace(np.linalg.inv(M_plan) @ M)

    def __crit_tau(self, tau, a, plan, th):
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

    def rand_plan(self, N, q=None, bounds=None):
        r = self.__r
        s = len(self.__th)
        if q is None:
            q = int((s + 1) * s / 2 + 1)
        x = np.random.uniform(-1, 1, [q, r, N])
        p = [1 / q] * q
        return [x, p]

    def dual(self, plan, th=None, d=0.05):
        ''' plan '''
        dmu = autograd.grad(self.__mu)  # this is *not* time consuming

        plan = copy.deepcopy(plan)

        if th is None:
            th = self.__th
        else:
            th = np.array(th)

        eta = len(th)
        r = self.__r
        X, p = plan  # TODO: make plan class
        N = X.shape[-1]

        crit_tau_grad = autograd.grad(self.__crit_tau)

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
                    return list(plan)

                if mu > eta:
                    break

            while True:
                # XXX: this was needed to get non singular tau value,
                # not sure if it is still needed
                tau_guess = np.random.uniform(size=1)
                tau_crit = self.__crit_tau(tau_guess, x_opt,
                                         copy.deepcopy(plan), th)
                if not np.isnan(tau_crit):
                    break

            rez = scipy.optimize.minimize(fun=self.__crit_tau, x0=tau_guess,
                                          args=(x_opt, copy.deepcopy(plan),
                                                th),
                                          bounds=[(0, 1)],
                                          method='SLSQP', jac=crit_tau_grad)

            tau_opt = rez['x']

            # add x_opt, tau_opt to plan
            X, p = plan
            x_opt = np.expand_dims(x_opt, 0)
            X = np.concatenate([X, x_opt])
            tau_opt = tau_opt[0]
            p = [p_i - tau_opt / len(p) for p_i in p]
            p.append(tau_opt)
            plan = [X, p]

            # clean plan
            plan = self.clean(copy.deepcopy(plan))

            # continue
