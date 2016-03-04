import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
np.random.seed(0)
num_sample, num_class, iteration = 10000, 5, 600
mu,sigma =  np.arange(num_class), 0.2*np.ones(num_class)
alpha = np.random.rand(num_class)
alpha = alpha / np.sum(alpha)
class_sample=np.zeros(num_class)
#num_sample = np.sum(each_class_sample)
for k in range(num_class):
    class_sample[k]=int(num_sample*alpha[k])
    if(k==0): x=sigma[k]*np.random.randn(class_sample[k])+mu[k]
    else:x = np.append(x,sigma[k]*np.random.randn(class_sample[k]) + mu[k])
total_sample=int(np.sum(class_sample))
mu_est, sigma_est = np.random.uniform(0,1,num_class), np.random.uniform(0,1,num_class)
alpha_est = np.ones(num_class)/num_class
#mu_est, sigma_est = np.ones(num_class), np.ones(num_class)
#alpha_est = np.ones(num_class)/num_class
p = np.zeros((total_sample,num_class), dtype=np.float)
ganma=np.zeros((total_sample,num_class),dtype=np.float)
num_each_sample_est = np.zeros(num_class)
x2=x*x
epc=0.00001
for t in range(iteration):
    #E-step
    for k1 in range(num_class):
        #print(mu_est[k1], sigma_est[k1], alpha_est[k1])
        mu_est_vec=mu_est[k1]*np.ones(len(x))
        p[:,k1] = alpha_est[k1] * np.exp( -(x2 -2.0*mu_est[k1]*x+mu_est[k1]**2) * 1.0/(2.0*sigma_est[k1]**2))/(np.sqrt(2.0*np.pi)*sigma_est[k1])
    for n in range(total_sample):
        sum_p_of_n=np.sum(p[n,:])
        if(epc<=sum_p_of_n):ganma[n,:]=p[n,:]/sum_p_of_n
        else: ganma[n,:]=np.zeros((1,num_class),dtype=np.float)
    #M-step
    for k2 in range(num_class):
        num_each_sample_est[k2] = int(np.sum( ganma[:,k2])) # this is correspond to sums up all of gannma_k by x
        mu_est[k2] = np.dot(ganma[:,k2],x) / num_each_sample_est[k2]
        sigma_est[k2] = np.sqrt( np.dot( ganma[:,k2], (x-mu_est[k2])**2 )/num_each_sample_est[k2] )
        alpha_est[k2] = num_each_sample_est[k2] / num_sample
        alpha_sum=np.sum(alpha_est)
        for k3 in range(num_class):
            alpha_est[k]/=alpha_sum

print("#mu = ",mu,"\n#mu_est = ",mu_est)
print("#sigma = ",sigma,"\n#sigma_est = ",sigma_est)
print("#alpha = ",alpha, "\n#alpha_est = ", alpha_est)
# plot
num_bin = 50
hist, bins = np.histogram(x,bins=num_bin)
for i in range(num_class):
    plt.plot(bins, mlab.normpdf(bins, mu_est[i], sigma_est[i]), 'r--')
width = 0.7 * (bins[1]-bins[0])
center = (bins[:-1]+ bins[1:])/2
plt.bar(center,hist/(2*num_sample/num_bin),align='center',width=width)
plt.show()


