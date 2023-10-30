"""
This module provides example of least square method example
"""

import random as rd
import numpy as np
import matplotlib.pyplot as plt

# set the idea R
IDEA_R = 8
# set the idea measure based on idea R
idea_i_measures = [rd.uniform(0.1, 0.8) for i in range(5)]
idea_v_measures = [i * IDEA_R for i in idea_i_measures]

# add measure noise
v_measures = np.array([v + rd.uniform(-0.0, 0.0) for v in idea_v_measures])
i_measures = np.array([i + rd.uniform(-0.12, 0.12) for i in idea_i_measures])

# using least square function
y = np.array([v_measures]).T
H = np.array([i_measures]).T

x_hat = np.linalg.inv(H.T @ H) @ H.T @ y
print(f"x_hat {x_hat}")

I_line = np.arange(0, 0.8, 0.1).reshape(8, 1)
V_line = IDEA_R * I_line

plt.scatter(i_measures, v_measures)
plt.plot(I_line, V_line)
plt.xlabel("Current (A)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.show()
