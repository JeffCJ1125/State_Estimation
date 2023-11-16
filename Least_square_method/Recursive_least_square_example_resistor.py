"""
This module provides example of Recursive Least Square method
"""

import math
import random as rd
import numpy as np
import matplotlib.pyplot as plt

USE_RANDOM = False

# measurement process noise
VARIANCE = 0.001
if USE_RANDOM:
    # set the idea R
    IDEA_R = 8
    # set the idea measure based on idea R
    idea_i_measures = [rd.uniform(0.1, 0.8) for i in range(8)]
    idea_v_measures = [i * IDEA_R for i in idea_i_measures]

    # add measure noise
    v_measures = np.array(
        [v + rd.gauss(0, math.sqrt(VARIANCE)) for v in idea_v_measures]
    )
    i_measures = np.array(
        [i + rd.gauss(0, math.sqrt(VARIANCE)) for i in idea_i_measures]
    )
else:
    v_measures = np.array(
        [
            0.96175243,
            1.59703192,
            6.0576919,
            1.1021352,
            3.84036949,
            1.33292009,
            5.98216595,
            4.24744803,
        ]
    )
    i_measures = np.array(
        [
            0.11865363,
            0.23525343,
            0.82637064,
            0.14504437,
            0.46592422,
            0.1606097,
            0.76730948,
            0.52378078,
        ]
    )

print(f"input measure v {v_measures}")
print(f"input measure i {i_measures}")
# initial quess for the state. since we are recursive measurement,
# we use same multimeters to measure resistor.
x_list = []
p_list = []
x_0 = np.array([[7]])
p_0 = np.array([[0.1]])
R = np.array([[VARIANCE]])

x_list.append(x_0)
p_list.append(p_0)

for i in range(1, len(v_measures) + 1):
    # set up measurement y_t
    y_i = np.array([[v_measures[i - 1]]])
    H_i = np.array([[i_measures[i - 1]]])
    # update estimate state and gain.
    p_last = p_list[-1]
    x_last = x_list[-1]
    K_i = p_last @ H_i.T @ np.linalg.inv(H_i @ p_last @ H_i.T + R)
    x_i = x_last + K_i @ (y_i - H_i @ x_last)
    I = np.eye(1)
    print(I)
    p_i = (I - K_i @ H_i) @ p_last
    print(f"x{i} {x_i}")
    x_list.append(x_i)
    p_list.append(p_i)

plt.scatter(i_measures, v_measures, label="Data")
plt.xlabel("Current (A)")
plt.ylabel("Voltage (V)")
plt.grid(True)

I_line = np.arange(0, 0.9, 0.1).reshape(9, 1)

for k in range(i_measures.shape[0]):
    V_line = x_list[k][0, 0] * I_line
    plt.plot(I_line, V_line, label=f"Measurement {k}")

plt.legend()
plt.show()
