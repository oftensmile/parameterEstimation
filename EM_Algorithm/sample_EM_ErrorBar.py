# make arbitrary K  class mixured gauss distribution,
# Error Bar (class = k) = +- sigma_k/sqrt(N_k), sigma ~ sigma_true
import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
np.random.seed(0)
num_sample, num_class, iteration = 1000, 10, 50
mu,sigma = 3 * np.arange(num_class), np.ones(num_class)
alpha = np.random.rand(num_class)
alpha = alpha / np.sum(alpha)
each_class_sample = np.array((num_sample * alpha),dtype=np.int)#各クラスのサンプル数が記憶される
num_sample = np.sum(each_class_sample)
x = []
for k in range(num_class):
    x = np.append([x],[ sigma[k]*np.random.randn(each_class_sample[k]) + mu[k]])
mu_est, sigma_est = np.ones(num_class), np.ones(num_class)
alpha_est = np.ones(num_class)
p = np.zeros((num_sample,num_class), dtype=np.float)
num_each_sample_est = np.zeros(num_class)

for t in range(iteration):
    #E-step
    x2 = x*x
    for k in range(num_class):
        p[:,k] = alpha_est[k] * np.exp( -(x2 -2 * x *mu[k] +mu[k]**2)  )/sigma_est[k]
    for n in range(num_sample):
        p[n,:] = p[n,:]/np.sum(p[n,:])
    for k in range(num_class):
        num_each_sample_est[k] = int(np.sum( p[:,k] ))
    #M-step
    for k in range(num_class):
        mu_est[k] = np.dot(p[:,k],x) / num_each_sample_est[k]
        sigma_est[k] = np.dot( p[:,k], (x-mu_est[k])**2 ) / num_each_sample_est[k]
        alpha_est[k] = num_each_sample_est[k] / num_sample
# result
error_barr = np.array( np.sqrt(sigma / each_class_sample ) )
print("#mu = ",mu,"\n#mu_est = ",mu_est,"\n±",error_barr)
print("#sigma = ",sigma,"\n#sigma_est = ",sigma_est)
print("#alpha = ",alpha, "\n#alpha_est = ", alpha_est)

ebar_file, mu_est_file= open('./error_bar.dat','w' ), open('mu_est.dat', 'w')
for k in range(num_class):
    ebar_file.write(str(k+1) +  ' ' + str(mu[k]) + ' ' +str(error_barr[k]) +'\n')
    mu_est_file.write(str(k+1) + ' ' + str(mu_est[k]) + '\n')
#e_bar.write(str(k+1)+'\n')
ebar_file.close()
mu_est_file.close()

# plot
num_bin = 50
hist, bins = np.histogram(x,bins=num_bin)
for i in range(num_class):
    plt.plot(bins, mlab.normpdf(bins, mu_est[i], sigma_est[i]), 'r--')
width = 0.7 * (bins[1]-bins[0])
center = (bins[:-1]+ bins[1:])/2
plt.bar(center,hist/(2*num_sample/num_bin),align='center',width=width)
plt.show()

