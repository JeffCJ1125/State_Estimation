"""
This module provides an example of Kalman filter to estimate the Population of Wombat
This example is the Computer exercises in Dan Simon, Optimal State Estimation (2006).
"""
import numpy as np
import matplotlib.pyplot as plt

# From one time step to the next, half of the existing wombat population dies,
#  but the number of new wombats is added to the population is equal to twice the food supply.
# P = 0.5 * last_P + 2 * F_k
# The food supply is constant except for zero-mean random fluctuations with a variance of 10.
# F = meanF +　w_k. w_k with variance of 10
# At each time step the wombat population is counted with an error that has zero mean and a variance of 10.
# P_c = TrueP + v_k. v_k with variance 10.

# the real state at time 0
REAL_F = 250
REAL_P = 650

# the initial estimate at time 0
X0 = np.array([[600], [200]])

P_CORVARIANCE = 500
F_CORVARIANCE = 200

P0 = np.array([[P_CORVARIANCE, 0], [0, F_CORVARIANCE]])
# xk = F_k @ last_k + G_k @ u_k + w_k
Q = np.array([[0, 0], [0, 10]])
R = np.array([[10]])


rp_list = []
rp_list.append(REAL_P)
rf_list = []
rf_list.append(REAL_F)

ep_list = []
ep_list.append(X0[0, 0])
ef_list = []
ef_list.append(X0[1, 0])
P_list = []
P_list.append(P0)

std_list = []
std_list.append(np.sqrt(P_CORVARIANCE))
f_error_list = []
f_error_list.append(REAL_F - X0[1, 0])
for i in range(1, 1001):
    rf_k = REAL_F + np.random.normal(0, 10)
    rp_k = rp_list[-1] * 0.5 + 2 * rf_k
    rp_list.append(rp_k)
    rf_list.append(rf_k)

    # prediction step
    last_p = ep_list[-1]
    last_f = ef_list[-1]
    last_P = P_list[-1]

    F_k = np.array([[0.5, 2], [0, 1]])
    last_x = np.array([[last_p], [last_f]])

    x_k = F_k @ last_x
    P_k = F_k @ last_P @ F_k.T + Q
    # print(f"x_k shape {x_k.shape}")
    # print(f"P_k shape {P_k.shape}")
    # Measurement update
    H_k = np.array([[1, 0]])
    # print(f"H_k shape {H_k.shape}")
    # print(f"R shape {R.shape}")
    K_k = P_k @ H_k.T @ np.linalg.inv(H_k @ P_k @ H_k.T + R)

    y_k = rp_k + np.random.normal(0, 10)

    x_k = x_k + K_k @ (y_k - H_k @ x_k)
    P_k = (np.eye(2) - K_k @ H_k) @ P_k

    ep_list.append(x_k[0, 0])
    ef_list.append(x_k[1, 0])

    f_error_list.append(rf_list[-1] - ef_list[-1])

    std_list.append(np.sqrt(P_k[0, 0]))
    P_list.append(P_k)
    # print(f"estimate p {ep_list[-1]:.3f} f {ef_list[-1]:.3f}")
    # print(f"true p {rp_list[-1]:.3f} f {rf_list[-1]:.3f}")
print(len(ep_list))
# print(len())
# plt.scatter(range(0,11), ep_list)
# plt.scatter(range(0,11), rp_list, c="red")
plt.title("Population")
plt.plot(range(0, len(ep_list)), ep_list)  # '-o' 指定線條和點的風格
plt.plot(range(0, len(rp_list)), rp_list)
# plt.grid(True)
plt.show()

# ax2 = plt.gca().twinx()
plt.title("food supply")
plt.plot(range(0, len(ef_list)), ef_list, "-o", color="green")
plt.plot(range(0, len(rf_list)), rf_list, "-o", color="purple")
# plt.grid(True)
plt.show()


plt.title("standard deviation of the population")
plt.plot(range(0, len(std_list)), std_list, "-o", color="green")
plt.grid(True)
plt.show()


plt.title("food supplyestimation error")
plt.plot(range(0, len(f_error_list)), f_error_list, "-o", color="green")
plt.grid(True)
plt.show()
print(np.std(ef_list))
