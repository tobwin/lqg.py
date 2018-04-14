import numpy as np


# linear quadratic estimator (Kalman filter)
def lqe(A, C, V, W, X0):
    horizon = len(A)
    for t in range(horizon):
        V[t] = V[t] * V[t].T
        W[t] = W[t] * W[t].T
    X0 = [X0[0] * X0[0].T]
    X1 = []
    K = []
    for t in range(horizon):
        K.append(X0[t] * C[t].T * np.linalg.pinv(C[t] * X0[t] * C[t].T + W[t]))
        X1.append(X0[t] - K[t] * C[t] * X0[t])
        if t < horizon-1:
            X0.append(A[t] * X1[t] * A[t].T + V[t])
    return K, X0, X1


# linear quadratic regulator
def lqr(A, B, Q, R):
    horizon = len(A)
    L = [np.matrix(np.zeros((2, 2)))]
    P = [Q[horizon - 1]]
    for t in reversed(range(horizon)):
        L.append(-np.linalg.pinv(R[t] + B[t].T * P[horizon - (t + 1)] * B[t]) * B[t].T * P[horizon - (t + 1)] * A[t])
        if t > 0:
            P.append(A[t].T * P[horizon - (t + 1)] * (A[t] + B[t] * L[horizon - t]) + Q[t])
    return list(reversed(L)), list(reversed(P))


# transforms time-invariant into constant time-varying (list of) matrices
def tvar(var, horizon, idx=None):
    if not type(var) is list:
        temp = []
        if not idx:
            for t in range(horizon):
                temp.append(var)
        else:
            for t in range(horizon):
                temp.append(np.matrix(np.zeros(np.shape(var))))
            temp[idx] = var
        var = temp
    return var


# main LQG class definition
class LQG:
    def __init__(self, horizon):
        self.horizon = horizon
        self.var = {'A': None, 'B': None, 'C': None,
                    'P': None, 'Q': None, 'R': None,
                    'V': None, 'W': None,
                    'X0': None, 'X1': None}
    
    def define(self, string, val):
        for char in string:
            if char in 'ABCRVW':
                self.var[char] = tvar(val, self.horizon)
            elif char == 'Q':
                self.var['Q'] = tvar(val, self.horizon, -1)
            elif char == 'X':
                self.var['X0'] = tvar(val, 1)
    
    def kalman(self):
        self.var['K'], self.var['X0'], self.var['X1'] = \
            lqe(self.var['A'], self.var['C'], self.var['V'], self.var['W'], self.var['X0'])
    
    def control(self):
        self.var['L'], self.var['P'] = \
            lqr(self.var['A'], self.var['B'], self.var['Q'], self.var['R'])
    
    def sample(self, n=1, x0=None, x=None, u=None, v=None, w=None):
        self.kalman()
        self.control()

        a = np.shape(self.var['A'][0])
        c = np.shape(self.var['C'][0])

        if x0 is None:  # default zero
            x0 = [np.matrix(np.zeros((a[0], n)))]
        elif min(np.shape(x0)) == 1 or type(x0) is list:  # provided mean
            x0 = np.reshape(np.matrix(x0), (max(np.shape(x0)), 1))
            x0 = [np.tile(x0, (1, n))]
        assert np.shape(x0[0]) == (a[0], n)

        if x is None:  # sample initial error
            e = np.random.randn(a[0], n)
            x = [x0[0] + self.var['X0'][0] * e]
        else:  # provided initial states
            assert np.shape(x) == (a[0], n)
            x = [x]
            e = np.linalg.pinv(self.var['X0'][0]) * (x[0] - x0[0])

        if w is None:
            w = [np.random.randn(c[0], n)]
            for t in range(self.horizon - 1):
                w.append(np.random.randn(c[0], n))
        if v is None:
            v = [np.random.randn(a[0], n)]
            for t in range(self.horizon - 1):
                v.append(np.random.randn(a[0], n))

        y = [self.var['C'][0] * x[0] + self.var['W'][0] * w[0]]
        x1 = [x0[0] + self.var['K'][0] * (y[0] - self.var['C'][0] * x0[0])]
        if u is None:
            u = [self.var['L'][0] * x1[0]]
        for t in range(self.horizon-1):
            x0.append(self.var['A'][t] * x1[t] + self.var['B'][t] * u[t])
            x.append(self.var['A'][t] * x[t] + self.var['B'][t] * u[t] + self.var['V'][t] * v[t])
            y.append(self.var['C'][t] * x[t] + self.var['W'][t] * w[t])
            x1.append(x0[t] + self.var['K'][t] * (y[t] - self.var['C'][t] * x0[t]))
            if len(u) <= t+1:
                u.append(self.var['L'][t] * x1[t])

        noise = {'x': e, 'v': v, 'w': w}
        kf = {'x0': x0, 'x1': x1}
        data = {'x': x, 'y': y, 'u': u, 'kf': kf, 'noise': noise, 'cost': {}}
        return data
