import numpy as np
d=5
class Matrices():
    def __init__(self,mat):
        self.mat=mat
        #self.mat=np.matrix(mat)
"""
def build_matrix(n):
    global matrices
    matrices=tuple(Matrices(np.ones((i+1,i+1))) for i in range(n) )
"""
def build_matrix(x=[[]]):
    global matrices
    matrices=tuple(Matrices(x) for i in range(1))
    #matrices=tuple(Matrices(np.ones((i+1,i+1))) for i in range(n) )
#build_matrix(3)

build_matrix(np.ones((3,3)))

print(np.shape(matrices))
for M in matrices:
    print(M.mat,"\n")

build_matrix(np.ones((3,3)))
matrices2=matrices
for i in range(2,10):
    matrices2=np.append(matrices2,Matrices(i*np.ones((3,3))) )
for M in matrices2:
    print(M.mat,"\n")

temp=np.random.choice(matrices2)
print("\n",temp.mat,"\n")
