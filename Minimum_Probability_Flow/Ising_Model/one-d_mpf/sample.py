import numpy as np
N=1000
rot=5
for i in range(N):
    print np.cos(i*(rot*3.14/N)),np.sin(i*(rot*3.14/N)),i
