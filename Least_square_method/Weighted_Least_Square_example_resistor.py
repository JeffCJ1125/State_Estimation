import numpy as np
import math
import random as rd
import matplotlib.pyplot as plt

# set the idea R
ideaR = 5
# set the idea measure based on idea R
idea_i_measures = [rd.uniform(0.1, 0.8) for i in range(8)]
idea_v_measures = [i * ideaR for i in idea_i_measures]

# add bad multimeter noise based on variance
bad_variance = 0.1
v_measures_bad = np.array(
    [v + rd.gauss(0, math.sqrt(bad_variance)) for v in idea_v_measures[:4]]
)
i_measures_bad = np.array(
    [i + rd.gauss(0, math.sqrt(bad_variance)) for i in idea_i_measures[:4]]
)

# add good multimeter noise based on variance
good_variance = 0.001
v_measures_good = np.array(
    [v + rd.gauss(0, math.sqrt(good_variance)) for v in idea_v_measures[4:]]
)
i_measures_good = np.array(
    [i + rd.gauss(0, math.sqrt(good_variance)) for i in idea_i_measures[4:]]
)

v_measures = np.concatenate((v_measures_bad, v_measures_good))
i_measures = np.concatenate((i_measures_bad, i_measures_good))

# using least square function
y = np.array([v_measures]).T
H = np.array([i_measures]).T

# define R matrix
R = np.zeros([8, 8])
R[:4, :4] = np.eye(4) * bad_variance
R[4:, 4:] = np.eye(4) * good_variance

invR = np.linalg.inv(R)
# print(f"R = \n {R}")

x_hat = np.linalg.inv(H.T @ invR @ H) @ H.T @ invR @ y
print(f"x_hat {x_hat}")

I_line = np.arange(0, 0.8, 0.1).reshape(8, 1)
V_line = ideaR * I_line

plt.scatter(i_measures_bad, v_measures_bad)
plt.scatter(i_measures_good, v_measures_good, c="red")
plt.plot(I_line, V_line)
plt.xlabel("Current (A)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.show()
