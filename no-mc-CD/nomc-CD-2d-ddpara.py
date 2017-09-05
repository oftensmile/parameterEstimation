#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt

#np.random.seed(1)
t_interval = 100
#parameter ( System )
d, N_sample,N_model = 6,100,30 #124, 1000
N_remove=100
#parameter ( MPF+GD )
lr,eps =1.0, 1.0e-100
t_gd_max=1000 

class Matrices():
    def __init__(self,mat):
        self.mat=mat

def build_matrix(x=[[]]):
    global matrices
    matrices=tuple(Matrices(x) for i in range(1))
 
def gen_mcmc(J1=[[]],J2=[[]],x=[[]]):
    for i1 in range(d):
        for i2 in range(d):
            #Heat Bath
            diff_E=2.0*x[i1][i2]*(J1[(i1+d-1)%d][i2]*x[(i1+d-1)%d][i2]+J1[i1][i2]*x[(i1+1)%d][i2]
                    +J2[i1][(i2+d-1)%d]*x[i1][(i2+d-1)%d]+J2[i1][i2]*x[i1][(i2+1)%d])#E_new-E_old
            r=1.0/(1+np.exp(diff_E)) 
            R=np.random.uniform(0,1)
            if(R<=r):
                x[i1][i2]=x[i1][i2]*(-1)
    return x

if __name__ == '__main__':
    x=np.random.choice([-1,1],(d,d))
    #Target Structure of the graph
    # vertical parameter
    J1_data=[[0,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,1,0,0],
            [0,0,0,1,0,0],
            [0,1,0,1,0,0],
            [0,0,0,0,0,0]]
    # horizontal parameter
    J2_data=[[0,0,0,0,0,0],
            [0,1,0,1,1,0],
            [0,1,0,0,1,1],
            [0,1,1,0,0,0],
            [0,0,0,1,1,0],
            [0,0,0,0,0,0]]
    J1_model=np.ones((d,d))
    J2_model=np.ones((d,d))
    #C1_data=np.zeros((d,d))
    #C2_data=np.zeros((d,d))
    for n in range(N_sample+N_remove):
        if(n<N_remove):
            x=np.copy(gen_mcmc(J1_data,J2_data,x))
        elif(n==N_remove):
            x=np.copy(gen_mcmc(J1_data,J2_data,x))
            build_matrix(x)
            data_matrix=matrices
        else:
            for t in range(t_interval):
                x=np.copy(gen_mcmc(J1_data,J2_data,x))
            data_matrix=np.append(data_matrix,Matrices(x))

    for t_gd in range(t_gd_max):
        #C1_model=np.zeros((d,d))
        #C2_model=np.zeros((d,d))
        diff_expect1=np.zeros((d,d))
        diff_expect2=np.zeros((d,d))
        for M in data_matrix:
            x=np.copy(M.mat)
            for i1 in range(d):
                for i2 in range(d):
                    diff_E=2.0*x[i1][i2]*(J1_model[(i1+d-1)%d][i2]*x[(i1+d-1)%d][i2]+J1_model[i1][i2]*x[(i1+1)%d][i2]
                    +J2_model[i1][(i2+d-1)%d]*x[i1][(i2+d-1)%d]+J2_model[i1][i2]*x[i1][(i2+1)%d])#E_new-E_old

                    diff_expect1[i1][i2]+=(-2.0*x[i1][i2]*x[(i1+1)%d][i2])/((1+np.exp(diff_E))*N_sample*d**2)
                    diff_expect1[(i1+d-1)%d][i2]+=(-2.0*x[i1][i2]*x[(i1+d-1)%d][i2])/((1+np.exp(diff_E))*N_sample*d**2)
                    diff_expect2[i1][i2]+=(-2.0*x[i1][i2]*x[i1][(i2+1)%d])/((1+np.exp(diff_E))*N_sample*d**2)
                    diff_expect2[i1][(i2+d-1)%d]+=(-2.0*x[i1][i2]*x[i1][(i2+d-1)%d])/((1+np.exp(diff_E))*N_sample*d**2)
                    
        J1_model=J1_model - lr * diff_expect1
        J2_model=J2_model - lr * diff_expect2
        print(sum(sum(abs(J1_model-J1_data))), sum(sum(abs(J2_model-J2_data))) )



    #---------------- visualize -------------------#

    #Matrix to array
    J_true_vec1 = np.zeros(d*d)
    J_true_vec2 = np.zeros(d*d)
    J_model_vec1 = np.zeros(d*d)
    J_model_vec2 = np.zeros(d*d)
    for i in range(d):
        for j in range(d):
            J_true_vec1[d*i+j] = J1_data[i][j]   
            J_true_vec2[d*i+j] = J2_data[i][j]   
            J_model_vec1[d*i+j] = J1_model[i][j] 
            J_model_vec2[d*i+j] = J2_model[i][j] 

    #----- scatter plot -----#
    x_lin = np.linspace(-1,1,100)
    y_lin = np.linspace(-1,1,100)

    f1 = plt.figure(1)
    ax1 = f1.add_subplot(1,2,1)
    ax1.plot(x_lin,y_lin)
    ax1.scatter(J_true_vec1,J1_model)
    ax1.set_xlabel("J1_{true}")
    ax1.set_ylabel("J1_{model}")
    ax1.grid(True)

    ax1 = f1.add_subplot(1,2,2)
    ax1.plot(x_lin,y_lin)
    ax1.scatter(J_true_vec2,J2_model)
    ax1.set_xlabel("J2_{true}")
    ax1.set_ylabel("J2_{model}")
    ax1.grid(True)
    f1.savefig("true-vs-model.png")
    f1.show()

    #----- heat map -----#
    f2 = plt.figure(2)

    ax2 = f2.add_subplot(141)
    ax2.imshow(J1_data, interpolation="nearest")
    #ax2.set_colorbar()
    ax2.set_title("J1_{true}")

    ax2 = f2.add_subplot(142)
    ax2.imshow(J1_model, interpolation="nearest")
    #ax2.set_colorbar()
    ax2.set_title("J_{model}")

    ax2 = f2.add_subplot(143)
    ax2.imshow(J2_data, interpolation="nearest")
    #ax2.set_colorbar()
    ax2.set_title("J_{true}")

    ax2 = f2.add_subplot(144)
    ax2.imshow(J2_model, interpolation="nearest")
    #ax2.set_colorbar()
    ax2.set_title("J_{model}")

    f2.savefig("true-vs-model-heatmap.png")
    f2.show()
