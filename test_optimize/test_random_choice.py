import numpy as np
import sys
import matplotlib.pyplot as plt

N=100000
n=8
hist_num=np.zeros(n)
numbers=np.arange(n)
for i in range(N):
    index=np.random.choice(numbers)
    hist_num[index]+=1
print(hist_num)
for i in range(n):
    print(hist_num[i])
