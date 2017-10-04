
import control
import autograd.numpy as np
import autograd
import scipy
import itertools

# changing model structure can be faster. just not redefine the whole graph


# discrete
class Model(object):

    # TODO: introduce some more default argument values, check types, cast if
    # neccessary
    def __init__(self, F, C, G, H, x0_mean, x0_cov, w_cov, v_cov, th):
        """
        Arguments are all callables (functions) of 'th' returning python lists
        except for 'th' itself (of course, which is list itself)
        """

        # TODO: evaluate and cast everything to numpy matrices first
        # TODO: cast floats, ints to numpy matrices
        # TODO: allow both constant matrices and callables

        def wrap_np(f):
            return lambda th: np.array(f(th), ndmin=2)

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
        # FIXME: exception if among those there is a scalar

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
        F * x + C * u + G * w
        H * x + v

        # check controllability, stability, observability
        self.__validate()

        # if the execution reached here, all is fine so
        # define corresponding computational tensorflow graphs
        # self.__define_observations_simulation()
        # self.__define_likelihood_computation()

    def __isObservable(self, th=None):
        if th is None:
            th = self.__th
        F = np.array(self.__F(th))
        C = np.array(self.__C(th))
        n = self.__n
        obsv_matrix = control.obsv(F, C)
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

    # FIXME: update, fix for discrete
    def __isStable(self, th=None):
        if th is None:
            th = self.__th
        F = np.array(self.__F(th))
        eigv = np.linalg.eigvals(F)
        real_parts = np.real(eigv)
        return np.all(real_parts < 0)

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

    def fim(self, u, x0=None, th=None):
        """
        'u' is 2d numpy array
        """
        if th is None:
            th = self.__th
        else:
            th = np.array(th)

        s = len(th)
        n = self.__n

        # TODO: reduce code, do not repeat yourself
        # make a list and loop through it

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

        # eval
        F, C, G, H, Q, R, X0, P0 = [f(th) for f in lst]

        if x0 is not None:
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
        Onn = np.zeros([n, n])

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
                OFdX0 = np.vstack([Onn, FdX0])

                # u[:,[k]] - get k-th column as column vector
                return F_ @ X0 + OFdX0 + C_A @ u[:, [0]]
            elif k > 0:
                return F_A @ X_Ap + C_A @ u[:, [k]]

        def Cf(i):
            i = i + 1
            O = [np.zeros([n, n])] * i
            O = np.hstack(O) if i else []
            C = np.hstack([O, np.eye(n)]) if i else np.eye(n)
            O = [np.zeros([n, n])] * (s-i)
            O = np.hstack(O) if s-i else []
            C = np.hstack([C, O]) if s-i else C
            return C

        u = np.array(u, ndmin=2)  # FIXME: add ndmin to other array construtors
        # exception is thrown here if u is a 1 d python list
        N = u.shape[1]

        for k in range(N):
            if k == 0:
                E_A = np.zeros([n*(s+1), n*(s+1)])
                X_Ap = X_Ap_f(None, None, u, k)
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
            AM = np.zeros([s, s])

            EXX = E_A + X_Ap @ t(X_Ap)

            C0 = Cf(0)

            # FIXME: autograd does not support +=
            for i, j in itertools.product(range(s), range(s)):
                S1 = Sp(C0 @ EXX @ t(C0) @ t(dH[j]) @ invB @ dH[i])
                S2 = Sp(C0 @ EXX @ t(Cf(j)) @ t(H) @ invB @ dH[i])
                S3 = Sp(Cf(i) @ EXX @ t(C0) @ t(dH[j]) @ invB @ H)
                S4 = Sp(Cf(i) @ EXX @ t(Cf(j)) @ t(H) @ invB @ H)
                S5 = 0.5 * Sp(dB[i] @ invB @ dB[j] @ invB)
                # XXX: autograd does not support element assigning
                AM[i, j] = S1 + S2 + S3 + S4 + S5

            M = M + AM

            # TODO: dont forget to update P, dP etc.
            Pe = Pu
            dPe = dPu

        return M


    def norm_fim(plan):
        pass

    def d_crit(plan):
        pass
