import numpy as np
import math
import time
from numba import jit, njit

l_1 = 0.065  # length of link 1 [m]
l_2 = 0.125  # length of link 2 [m]
l_3 = 0.16   # length of link 3 [m]
L = [l_1, l_2, l_3]

def decorator_timer(some_function):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = some_function(*args, **kwargs)
        end = time.time() - t1
        return result, end
    return wrapper

@decorator_timer
@njit
def ccd_inverse_kinematics(target, joints, max_iterations=100, tolerance=1e-3):
    def distance(a, b):
        return np.linalg.norm(a - b)

    def rotate_around_axis(point, axis_point, axis, angle):
        s, c = np.sin(angle), np.cos(angle)
        R = np.array([
            [c + (1 - c) * axis[0] * axis[0],           (1 - c) * axis[0] * axis[1] - s * axis[2],  (1 - c) * axis[0] * axis[2] + s * axis[1]],
            [(1 - c) * axis[1] * axis[0] + s * axis[2], c + (1 - c) * axis[1] * axis[1],            (1 - c) * axis[1] * axis[2] - s * axis[0]],
            [(1 - c) * axis[2] * axis[0] - s * axis[1], (1 - c) * axis[2] * axis[1] + s * axis[0],  c + (1 - c) * axis[2] * axis[2]]
        ])
        return np.dot(R, (point - axis_point)) + axis_point

    for _ in range(max_iterations):
        for i in range(len(joints) - 1, -1, -1):
            current_end = joints[-1]
            if distance(current_end, target) < tolerance:
                return joints

            joint_to_end = current_end - joints[i]
            joint_to_target = target - joints[i]

            axis = np.cross(joint_to_end, joint_to_target)
            if np.linalg.norm(axis) < 1e-6:
                continue
            axis = axis / np.linalg.norm(axis)

            angle = np.arccos(np.dot(joint_to_end, joint_to_target) / (distance(joint_to_end, np.zeros(3)) * distance(joint_to_target, np.zeros(3))))
            for j in range(i + 1, len(joints)):
                joints[j] = rotate_around_axis(joints[j], joints[i], axis, angle)

    return joints

@decorator_timer
def InverseKinematics(X, L):
    x, y, z = X
    L1, L2, L3 = L
    th = [0, 0, 0]

    LL = np.hypot(x, y)
    D = np.sqrt(z**2 + (LL - L1)**2)

    a = (L3**2 - L2**2 - D**2) / (-2 * D * L2)
    b = (D**2 - L2**2 - L3**2) / (-2 * L2 * L3)

    if x == 0:
        j1 = np.pi / 2 if y > 0 else -np.pi / 2
    else:
        j1 = np.arctan(y / x)

    if not np.isnan(j1):
        th[0] = j1

    if a < -1 or a > 1 or b < -1 or b > 1:
        print("Configuration is not possible due to arccos domain error")
        return th

    j2 = np.arctan(z / (LL - L1)) - np.arccos((L3**2 - L2**2 - D**2) / (-2 * D * L2))
    if not np.isnan(j2):
        th[1] = j2

    j3 = np.pi - np.arccos((D**2 - L2**2 - L3**2) / (-2 * L2 * L3))
    if not np.isnan(j3):
        th[2] = j3

    return th

while(True):
    target = np.array([0.2, 0.2, 0.1])
    ccd_result, ccd_time = ccd_inverse_kinematics(target, np.array([[0, 0, 0], [l_1, 0, 0], [l_1 + l_2, 0, 0], [l_1 + l_2 + l_3, 0, 0]]))
    analytic_result, analytic_time = InverseKinematics(target, L)

    print("CCD Inverse Kinematics Result:", ccd_result)
    print("CCD Time:", ccd_time)
    print("Analytic Inverse Kinematics Result:", analytic_result)
    print("Analytic Time:", analytic_time)
    
    time.sleep(1)

