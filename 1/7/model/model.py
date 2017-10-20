
import control
import autograd.numpy as np
import autograd
import scipy
import itertools
import copy

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
        # self.__validate()

        # if the execution reached here, all is fine so
        # define corresponding computational tensorflow graphs
        # self.__define_observations_simulation()
        # self.__define_likelihood_computation()

        self.__d_crit_to_opt_grad_f = autograd.grad(self.__d_crit_to_optimize)

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

    def __isStable(self, th=None):
        if th is None:
            th = self.__th
        F = np.array(self.__F(th))
        eigv = np.linalg.eigvals(F)
        abs_val = np.absolute(eigv)
        return np.all(abs_val < 1)

    def __validate_model(self, th=None):
        # FIXME: do not raise exceptions, print warnings
        # TODO: prove, print matrices and their characteristics
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
            raise Exception('invalid shape of 'u'')

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

    def norm_fim(self, plan, u, th=None):
        ''' plan: list of list of 'x0' and list of 'p' '''

        x, p = plan
        x = np.array(x, ndmin=2)
        p = np.array(p)

        # FIXME:
        # validate plan
        for x_i, p_i in zip(x, p):
            if len(x_i) != self.__n:
                raise Exception('invalid plan: len(x_i) != n')
        Mn = 0

        # extract p, x and compute fim for every point of the plan
        for x_i, p_i in zip(x, p):
            Mn += p_i * self.fim(u, x_i, th)  # TODO: compute in parallel

        return Mn

    # calls norm_fim and logdet it
    def d_opt_crit(self, plan, u, th=None):
        Mn = self.norm_fim(plan, u, th)
        sign, logdet = np.linalg.slogdet(Mn)
        return -logdet

    def d_crit_opt_grad(self, plan, u, th=None):
        # flatten plan
        x, p = plan
        x = np.array(x).flatten()
        p = np.array(p)
        q = len(p)
        plan = np.hstack([x, p])  # 1d array
        grad = self.__d_crit_to_opt_grad_f(plan, q, u, th)
        return grad

    def __d_crit_to_opt_grad(self, plan, q, u, th=None):
        plan = np.array(plan)
        grad = self.__d_crit_to_opt_grad_f(plan, q, u, th)
        return grad

    # this is wraps around self.d_opt_crit() above
    # for scipy.optimize.minimize
    def __d_crit_to_optimize(self, plan, q, u, th=None):
        # unflatten plan
        p = plan[-q:]
        x = plan[:-q]
        x = np.array_split(x, q)
        plan = [x, p]
        crit = self.d_opt_crit(plan, u, th)
        return crit

    def direct(self, plan0, u, th=None, use_grad=True):
        ''' plan0: list of: list (or 2d np.array) of 'x0' and list of 'p' '''
        n = self.__n

        x, p = plan0

        for x_i, p_i in zip(x, p):
            if len(x_i) != n:
                raise Exception('invalid plan: len(x_i) != n')

        q = len(p)

        x_bounds = [(-1, 1)] * q * n
        p_bounds = [(0, 1)] * q
        bounds = x_bounds + p_bounds

        def heq(x):
            p = x[-q:]
            return np.sum(p) - 1

        constraints = {'type': 'eq', 'fun': heq}

        flatten = lambda l: [item for sublist in l for item in sublist]

        # FIXME: if x and p are arrays, this will sum them
        x0 = flatten(x) + p

        # pass jacobian
        rez = scipy.optimize.minimize(fun=self.__d_crit_to_optimize, x0=x0,
                                      args=(q, u, th), method='SLSQP',
                                      constraints=constraints, bounds=bounds)
        new_plan = rez['x']

        pn = new_plan[-q:]
        xn = new_plan[:-q].reshape([q, n])
        new_plan = [xn, pn]

        # TODO: return loss and its jacobian values
        # return dictionary
        return new_plan

    def clean(self, plan, dn=0.5, dp=0.05):
        ''' plan = [x, p], x is 2d array, p is list '''
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

        return x, p

    def __mu(self, x, M_plan, u, th):
        M = self.fim(u=u, x0=x, th=th)
        return -np.trace(np.linalg.inv(M_plan) @ M)

    def crit_tau(self, tau, a, plan, u, th):
        x, p = plan
        x = np.vstack([x, a])
        p = [p_i * (1 - tau) for p_i in p]
        p.append(tau)
        plan = [x, p]
        crit = self.d_opt_crit(plan, u, th)
        return crit

    # FIXME: seems like some shit like variables scope collisions
    def dual(self, plan, u, th=None, d=0.05):
        ''' plan '''
        dmu = autograd.grad(self.__mu)  # this is *not* time consuming

        plan = copy.deepcopy(plan)

        if th is None:
            th = self.__th
        else:
            th = np.array(th)

        eta = len(th)
        n = self.__n
        X, p = plan

        crit_tau_grad = autograd.grad(self.crit_tau)

        x_bounds = [(-1, 1)] * n

        while True:
            M_plan = self.norm_fim(plan, u, th)

            while True:

                x_guess = np.random.uniform(-1, 1, n)
                x_guess = np.array([0, 0])

                # nlopts <- list(xtol_rel=1e-3, maxeval=1e3)
                # TODO: pass gradient
                rez = scipy.optimize.minimize(fun=self.__mu, x0=x_guess,
                                              args=(M_plan, u, th),
                                              method='SLSQP', jac=dmu,
                                              bounds=x_bounds, tol=None,
                                              options=None)
                x_opt = rez['x']
                mu = -rez['fun']

                if abs(mu - eta) <= d:
                    return plan, self.d_opt_crit(plan, u)

                if mu > eta:
                    break

            while True:
                # XXX: this was needed to get non singular tau value,
                # not sure if it is still needed
                tau_guess = np.random.uniform(size=1)
                tau_guess = 0.5
                tau_crit = self.crit_tau(tau_guess, x_opt, copy.deepcopy(plan),
                        u, th)
                if not np.isnan(tau_crit):
                    break

            rez = scipy.optimize.minimize(fun=self.crit_tau, x0=tau_guess,
                                          args=(x_opt, copy.deepcopy(plan),
                                              u, th), bounds=[(0, 1)],
                                          method='SLSQP', jac=crit_tau_grad)

            tau_opt = rez['x']

            # add x_opt, tau_opt to plan
            X, p = plan
            X = np.vstack([X, x_opt])
            tau_opt = tau_opt[0]
            p = [p_i - tau_opt / len(p) for p_i in p]
            p.append(tau_opt)
            plan = [X, p]

            # clean plan
            plan = self.clean(copy.deepcopy(plan))

            # continue
