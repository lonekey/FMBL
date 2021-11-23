import numpy as np
import time
def compute(codes: list):
    # start = time.time()
    length = [len(i) for i in codes]
    median = np.median(length)
    score = np.arctan(length/median)
    # end = time.time()
    # print("代码长度时间", end-start)
    # print(len(score))
    return score

# codes = [["a", "apple"], ["bad"], ["good"]]
# a = compute(codes)
# print(a)