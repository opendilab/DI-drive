import numpy as np
from enum import Enum

Orders = Enum("Order", "Follow_Lane Straight Right Left ChangelaneLeft ChangelaneRight")


def adapt_order(incoming_obs_command):
    if incoming_obs_command == 1:  # LEFT
        return Orders.Left.value
    if incoming_obs_command == 2:  # RIGHT
        return Orders.Right.value
    if incoming_obs_command == 3:  # STRAIGHT
        return Orders.Straight.value
    if incoming_obs_command == 4:  # FOLLOW_LANE
        return Orders.Follow_Lane.value
    if incoming_obs_command == 5:  # CHANGE_LANE_LEFT
        return Orders.ChangelaneLeft.value
    if incoming_obs_command == 6:  # CHANGE_LANE_RIGHT
        return Orders.ChangelaneRight.value


def compute_angle(vec1, vec2):
    if isinstance(vec1, np.ndarray) or isinstance(vec1, list):
        arr1 = np.array([vec1[0], vec1[1], 0])
        arr2 = np.array([vec2[0], vec2[1], 0])
    else:
        arr1 = np.array([vec1.x, vec1.y, 0])
        arr2 = np.array([vec2.x, vec2.y, 0])
    cosangle = min(1.0, arr1.dot(arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2)))
    angle = min(np.pi / 2, np.abs(np.arccos(cosangle)))
    return angle


def compute_point_line_dis(point_x, point_y, vec_x, vec_y, point2_x, point2_y):
    '''
    Compute the distance between the vehicle and the middle oflane
    '''
    b = np.sqrt(vec_x ** 2 + vec_y ** 2)
    a = abs(vec_x * point2_y - vec_y * point2_x - vec_x * point_y + vec_y * point_x)
    return a / b


def compute_cirle(point1, vec1, point2, vec2):
    '''
    Compute the position of the turning circle and turning radius
    '''
    if abs(vec1[1]) < 1e-4:
        vec1[1] = 1e-4
    if abs(vec2[1]) < 1e-4:
        vec2[1] = 1e-4
    k1 = -vec1[0] / (np.sign(vec1[1]) * max(abs(vec1[1]), 1e-4))
    k2 = -vec2[0] / (np.sign(vec2[1]) * max(abs(vec2[1]), 1e-4))
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    x = (k1 * x1 - k2 * x2 + y2 - y1) / (k1 - k2)
    y = k1 * (x - x1) + y1
    r = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    return x, y, r


def compute_speed(vec):
    return np.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)
