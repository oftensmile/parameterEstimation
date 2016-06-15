#2016/05/19
##############
#   H = J*sum(xixj), J in R^1
##############
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv
import scipy.misc
np.random.seed(0)
lena=scipy.misc.lena()
print("shape of image =",lena.shape)
width_lena=len(lena)
height_lena=len(lena[1])

#parameter ( MCMC )
#t_interval = 10
#parameter ( System )
d_x,d_y, N_sample =width_lena,height_lena,10 #124, 1000
#N_remove=100
#parameter ( MPF+GD )
lr,eps =1, 1.0e-100
t_gd_max=3
def main():
    x = np.random.uniform(-1,1,d_x*d_y)
    x = np.array(np.sign(x))
    #Generate samples, which just added rando normal on to the image of Lena
    mu, sigma= 0.0, 10.0
    lena_vec = np.reshape(lena, d_x*d_y)
    for n in range(N_sample):
        x=lena_vec+np.random.normal(mu,sigma,d_x*d_y)
        if(n==0):X_sample = np.copy(x)
        elif(n>0):X_sample=np.vstack((X_sample,np.copy(x)))
    #MPF
    x_init = X_sample[0]
    x_last= X_sample[N_sample-1]
    mat_init_x=np.reshape(x_init,(d_x,d_y))
    mat_last_x=np.reshape(x_last,(d_x,d_y))

    theta_model1,theta_model2=0.3, 0.2  #Initial Guess
    print("#diff_E diff_E1_nin diff_E2_nin")
    for t_gd in range(t_gd_max):
        gradK1,gradK2=0.0,0.0
        n_bach=len(X_sample)
        for nin in range(n_bach):
            x_nin=np.copy(X_sample[nin])
            gradK1_nin,gradK2_nin=0.0,0.0
            for ix in range(d_x):
                for iy in range(d_y):
                    #diff_E=E(x_new)-E(x_old)
                    diff_delE1_nin=x_nin[ix+iy*d_x]*(x[(ix+d_x-1)%d_x+iy*d_x]+x[(ix+1)%d_x+iy*d_x])
                    diff_delE2_nin=x_nin[ix+iy*d_x]*(x[ix+d_x*((iy+d_y-1)%d_y)]+x[ix+d_x*((iy+1)%d_y)])
                    diff_E1_nin=diff_delE1_nin*theta_model1
                    diff_E2_nin=diff_delE2_nin*theta_model2
                    diff_E_nin=diff_E1_nin+diff_E2_nin
                    #adhoc
                    diff_E_nin*=0.001
                    gradK1_nin+=diff_delE1_nin*np.exp(diff_E_nin)/(d_x*d_y)
                    gradK2_nin+=diff_delE2_nin*np.exp(diff_E_nin)/(d_x*d_y)
            gradK1+=gradK1_nin/n_bach
            gradK2+=gradK2_nin/n_bach
        theta_model1=theta_model1 - lr * gradK1
        theta_model2=theta_model2 - lr * gradK2
        theta_diff1=abs(theta_model1-J)
        theta_diff2=abs(theta_model2-J)
        print(t_gd,np.abs(gradK1),np.abs(gradK2),theta_diff1,theta_diff2)
    print("#theta1,=",J,"theta1,theta2 _estimated=",theta_model1,theta_model2)
    noize=np.random.randn(width_lena,height_lena)*30
    damegede=lena+noize
    #Plot of the 
    plt.figure()
    plt.subplot(221)
    plt.imshow(mat_init_x,cmap="gray",interpolation='nearest')
    plt.title("Image init")
    plt.subplot(222)
    plt.imshow(mat_last_x,cmap="gray",interpolation='nearest')
    plt.title("Image last")
    plt.subplot(223)
    plt.imshow(lena,cmap="gray",interpolation='nearest')
    plt.title("Lena")
    plt.subplot(224)
    plt.imshow(damegede,cmap="gray",interpolation='nearest')
    plt.title("dameged Lena")
    plt.show()


#Generate sample-dist
#J1,J2=0.01,0.01 # =theta_sample
J=0.001
if __name__ == '__main__':
    main()
    


