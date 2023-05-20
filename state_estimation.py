import time
import numpy as np
import math


class Estimator:
    def __init__(self, sysd, max_k, epsilon):
        self.on = True
        # optional, parameter p1
        self.Ad = sysd.A
        self.Bd = sysd.B
        self.m = np.size(self.Ad, 0)
        self.epsilon = epsilon
        self.max_k = max_k
        # initiate your data
        self.Ad_k = [np.eye(self.m)]
        for i in range(max_k):
            self.Ad_k.append(self.Ad_k[-1].dot(self.Ad))
        self.Ad_k_Bd = [i.dot(self.Bd) for i in self.Ad_k]
        self.Ad_k_Ad_k_T = [i.dot(i.T) for i in self.Ad_k]
        self.epsilon_coef = []
        sqrt_term = np.zeros((self.m, 1))
        for k in range(max_k):
            for i in range(self.m):
                sqrt_term[i, 0] += math.sqrt(self.Ad_k_Ad_k_T[k][i, i])
            self.epsilon_coef.append(sqrt_term.copy())
        # print(self.epsilon_coef)

    def estimate(self, x_a: np.array, control_lst: np.array, debug=True):
        start = time.time()
        k = np.size(control_lst, 1)
        assert k < self.max_k
        assert np.size(x_a, 0) == self.m
        control_sum_term = np.zeros((self.m, 1))
        for j in range(k):
            control_sum_term += self.Ad_k_Bd[j].dot(control_lst[:, k - 1 - j:k - j])
        x_0 = self.Ad_k[k].dot(x_a) + control_sum_term
        if debug:
            print('x_a=', x_a)
            print('control_lst=', control_lst)
            print('x_0=', x_0)
        e = np.ones((self.m, 1)) * self.epsilon * self.epsilon_coef[k]
        x_0_lo = x_0 - e
        x_0_up = x_0 + e
        end = time.time()
        if debug:
            print('Use', end - start, 'seconds for', k, 'steps')
        return x_0_lo, x_0_up

    def get_deadline(self, x_a, safe_set_lo, safe_set_up, control: np.array, max_k):
        k = max_k
        breaked = False
        # print('control', control)
        for i in range(max_k):
            i += 1
            control_series = control
            for j in range(i - 1):
                control_series = np.hstack((control_series, control))
            # print("control_ser=", control_series)
            x_0_lo, x_0_up = self.estimate(x_a, control_series, debug=False)
            for j in range(np.size(x_a, 0)):
                if safe_set_lo[j] < x_0_lo[j, 0] < safe_set_up[j] and safe_set_lo[j] < x_0_up[j, 0] < safe_set_up[j]:
                    pass
                else:
                    breaked = True
                    break
            if breaked:
                k = i - 1
                print('x_0_up', x_0_up)
                break
        return k

    def get_safetime(self, x_a, safe_set_lo, safe_set_up, control_list: np.array, max_k):
        assert max_k < self.max_k
        # only analyze max_k step
        num_control = np.size(control_list, 1)
        if num_control > max_k:
            control_list = control_list[:, :max_k]
            num_control = max_k
        print('num_control=', num_control)
        breaked = False
        k = max_k
        print('safe_set_lo=', safe_set_lo)
        print('safe_set_up=', safe_set_up)
        for i in range(num_control):
            i += 1
            control_series = control_list[:, :i]
            x_i_lo, x_i_up = self.estimate(x_a, control_series, debug=False)
            print('i=', i, 'x_lo=',x_i_lo,' x_up=', x_i_up)
            for j in range(np.size(x_a, 0)):
                if safe_set_lo[j] < x_i_lo[j, 0] < safe_set_up[j] and safe_set_lo[j] < x_i_up[j, 0] < safe_set_up[j]:
                    pass
                else:
                    breaked = True
                    break
            if breaked:
                k = i - 1
                print('x_0_up', x_i_up)
                break
        return k


if __name__ == "__main__":
    from recovery import recovery
    from control.matlab import ss, c2d

    A = [[-10, 1], [-0.02, -2]]
    B = [[0], [2]]
    C = [1, 0]
    sys = ss(A, B, C, 0)
    dt = 0.02
    sysd = c2d(sys, dt)

    initial_set_lo = [7.999902067622, 79.998780693465]
    initial_set_up = [7.999902067622887, 79.998780693465960]
    target_set_lo = [3.9, -100]
    target_set_up = [4.1, 100]
    safe_set_lo = [1, -150]
    safe_set_up = [8, 150]
    control_lo = [-150]
    control_up = [150]
    control_lst = recovery(sysd, 10, initial_set_lo, initial_set_up, target_set_lo, target_set_up, safe_set_lo,
                           safe_set_up,
                           control_lo, control_up)

    t = Estimator(sysd, 20, 1e-7)
    x0_lo, x0_up = t.estimate(np.array([[7.999902067622887], [79.998780693465960]]), control_lst)
    print(x0_lo, x0_up)
