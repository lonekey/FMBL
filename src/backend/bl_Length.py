import numpy as np
def compute(codes: list):
    length = [len(i) for i in codes]
    median = np.median(length)
    score = np.arctan(length/median)
    return score

# codes = [["a", "apple"], ["bad"], ["good"]]
# a = compute(codes)
# print(a)