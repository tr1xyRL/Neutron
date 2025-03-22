import numpy as np
from math import sqrt


def sign(x: float) -> int:
    return -1 if x < 0 else 1

def normalize(x: np.array) -> float:
    norm = np.linalg.norm(x)
    if norm == 0:
       return x
    return x / norm

def distance(x: np.array, y: np.array)->float:
    return sqrt(sum([val*val for val in x-y]))

def distance2D(x: np.array, y: np.array)->float:
    x[2] = 0
    y[2] = 0
    return distance(x, y)

def clamp(max_range: float, min_range: float, number: float) -> float:
    return max((min_range, min((number, max_range))))
    