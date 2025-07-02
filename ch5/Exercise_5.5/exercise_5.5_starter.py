import numpy as np
import math
def WMMSE_sum_rate(p_int, H, Pmax, var_noise):
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        vnew = vnew + math.log2(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        # Write your code here
            break

    p_opt = np.square(b)
    return p_opt
if __name__ == '__main__':
    P_max = 1
    P_ini = P_max*np.ones(4) # inital power
    H =[[1.616, 0.284, 0.704, 1.919],
        [0.699, 0.723, 0.636, 1.403],
        [0.150, 1.346, 1.006, 0.287],
        [0.778, 1.457, 1.215, 1.397]
    ]
    H = np.array(H) # The matrix of channel power gains
    var_noise = 1 # The noise power is assumed to 1
    Power_allocation = WMMSE_sum_rate(P_ini, H, P_max, var_noise)
    print(f"The power allocation for each D2D is {Power_allocation}.")
