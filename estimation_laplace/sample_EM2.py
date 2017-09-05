# make arbitrary K  class mixured gauss distribution,
import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
np.random.seed(0)
num_sample, num_class, iteration = 10000, 10, 500
mu,sigma = 3 * np.arange(num_class), np.ones(num_class)
alpha = np.random.rand(num_class)
alpha = alpha / np.sum(alpha)
class_sample=np.zeros(num_class)
#num_sample = np.sum(each_class_sample)
for k in range(num_class):
    class_sample[k]=int(num_sample*alpha[k])
    if(k==0): x=[sigma[k]*np.random.randn(class_sample[k])+mu[k]]
    else:x = np.append(x,[ sigma[k]*np.random.randn(class_sample[k]) + mu[k]])
total_sample=int(np.sum(class_sample))
mu_est, sigma_est = np.arange(num_class), np.arange(num_class)+1
alpha_est = np.random.randn(num_class)
p = np.zeros((total_sample,num_class), dtype=np.float)

q=np.zeros((num_sample),dtype=np.float)
num_each_sample_est = np.zeros(num_class)

for t in range(iteration):
    #E-step
    x2 = x*x
    for k in range(num_class):
        p[:,k] = alpha_est[k] * np.exp( -(x2 -2 * x *mu_est[k] +mu_est[k]**2)/(2.0*sigma_est[k]**2))/(np.sqrt(2.0*np.pi)*sigma_est[k])
    for n in range(total_sample):
        p[n,:] = p[n,:]*1.0/np.sum(p[n,:])
    for k in range(num_class):
        num_each_sample_est[k] = np.sum( p[:,k] )
    #M-step
    for k in range(num_class):
        mu_est[k] = np.dot(p[:,k],x) / num_each_sample_est[k]
        sigma_est[k] = np.dot( p[:,k], (x-mu_est[k])**2 ) / num_each_sample_est[k]
        alpha_est[k] = num_each_sample_est[k] / num_sample
# result
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
