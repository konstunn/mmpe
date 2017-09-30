
import control
import autograd.numpy as np
import autograd

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
            return lambda th: np.array(f(th))

        # store arguments, after that check them
        F = self.__F = wrap_np(F)
        C = self.__C = wrap_np(C)
        G = self.__G = wrap_np(G)
        H = self.__H = wrap_np(H)
        x0_mean = self.__x0_mean = wrap_np(x0_mean)
        x0_cov = self.__x0_cov = wrap_np(x0_cov)
        w_cov = self.__w_cov = wrap_np(w_cov)
        v_cov = self.__v_cov = wrap_np(v_cov)

        th = self.__th = np.array(th)

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
        x = np.random.multivariate_normal(x0_m.squeeze(), x0_cov)
        w = np.random.multivariate_normal(w_mean.squeeze(), w_cov)
        v = np.random.multivariate_normal(v_mean.squeeze(), v_cov)

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

    def fim(self, x0=None, u=None, th=None):
        """
        'u' is 2d numpy array
        """
        if th is None:
            th = self.__th
        else:
            th = np.array(th)

        s = len(th)

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

        # TODO: refactor
        jlst = [[np.squeeze(j) for j in np.dsplit(jel)] for jel in jlst]

        dF, dC, dG, dH, dQ, dR, dX0, dP0 = jlst

        # eval
        F, C, G, H, Q, R, X0, P0 = [f(th) for f in lst]

        if x0 is not None:
            X0 = x0

        C_A = np.vstack(dC)
        C_A = np.vstack((C, C_A))

        M = np.zeros([s, s])

        k = 0

        if k == 0:
            # E_A =
            pass
        elif k > 0:
            # E_A =
            pass

        pass

    def norm_fim(plan):
        pass

    def d_crit(plan):
        pass
