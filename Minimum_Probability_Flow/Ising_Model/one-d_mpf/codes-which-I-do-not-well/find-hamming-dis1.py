import numpy as np
import time 
import sys
np.random.seed(33)


def calc_E(x_tot=[[]],theta=[[]]):
    len_x=len(x_tot)
    E=np.zeros(len_x)
    for n in range(len_x):
        x_n=np.copy(x_tot[n])
        E[n]=np.matrix(x_n)*np.matrix(theta)*np.matrix(x_n).T
        print("E(n)=",E[n])
    return E
    
d=4
N_sample=6
theta=np.zeros((d,d))
for i in range(d):
    theta[i][(i+1)%d]=1
    theta[(i+1)%d][i]=1

for n in range(N_sample):
    if(n==0):
        X_sample=np.random.choice([-1,1],d)
    else:
        X_sample=np.vstack((X_sample,np.random.choice([-1,1],d)))
print("shape of X_sample = ",np.shape(X_sample))
print("X_sample=\n",X_sample)
dist_mat=d*np.ones((N_sample,N_sample))
dist_mat=2*np.copy(dist_mat)-2*(np.matrix(X_sample)*np.matrix(X_sample).T)
dist_mat/=4
print("distance=\n",dist_mat)
idx=np.where(dist_mat!=1)
print("idx=",idx)
dist_mat2=np.copy(dist_mat)
dist_mat2[idx]=0
print("dist_mat2=\n",dist_mat2)

E_vec=calc_E(X_sample,theta)
print("E_vec=",E_vec)
#   i,j-element = E_i - E_j
diff_E_mat=E_vec*dist_mat2-(E_vec*dist_mat2.T).T
print("diff_E_mat=\n",diff_E_mat)
exp_diff_E_mat=np.exp(diff_E_mat)
print("exp_diff_E_mat=\n",exp_diff_E_mat)
result=np.copy(exp_diff_E_mat)
#To sum up all exp_diff_E_mat's elements without exp(0).
idx=np.where(diff_E_mat!=0)
print("result[idx]=\n",result[idx])
#print("result=\n",result)
print("idx=",idx)
eliminame=np.sum(result[idx])
print("eliminame=",eliminame)

grad=np.zeros((d,d))
len_idx=len(idx[0])
for i in range(len_idx):
    print(i," (idx1,idx2) = (", idx[0][i],",",idx[1][i],")")
    diff_sample_pair_i=X_sample[idx[0][i]]-X_sample[idx[1][i]]
    print("diff_sample_pair_i=\n",diff_sample_pair_i)
    idx3=np.where(diff_sample_pair_i!=0)
    print("idx3=",idx3[0][0])
    temp_vec1=-X_sample[idx[1][i]][idx3]*X_sample[idx[1][i]]
    temp_vec1[idx3]=0
    print("check scalar=",(result[idx[0][i]][idx[1][i]] - result[idx[0][i]][idx[1][i]]**(-1) )
)
    grad[idx3]+=temp_vec1*(result[idx[0][i]][idx[1][i]] - result[idx[0][i]][idx[1][i]]**(-1) )
    grad.T[idx3]+=temp_vec1*(result[idx[0][i]][idx[1][i]] - result[idx[0][i]][idx[1][i]]**(-1) )
    print("grad=\n",grad)



#Cal partial theta 
