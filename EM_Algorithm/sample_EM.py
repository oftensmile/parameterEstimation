# make K=2  class of mixured gauss distribution,
import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
np.random.seed(0)
num_sample, num_class, iteration = 2000, 3, 50
mu,sigma = np.arange(num_class), np.ones(num_class)
mu[2] , mu[1] = -3,3
alpha = np.random.rand(num_class)
alpha = alpha / np.sum(alpha)
each_class_sample = np.array((num_sample * alpha),dtype=np.int)
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
    print(p)
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
print("#mu = ",mu," mu_est = ",mu_est)
print("#sigma = ",sigma," sigma_est = ",sigma_est)
print("#alpha = ",alpha, " alpha_est = ", alpha_est)

# plot
num_bin = 50
hist, bins = np.histogram(x,bins=num_bin)
y0 = mlab.normpdf(bins,mu_est[0],sigma_est[0])
y1 = mlab.normpdf(bins,mu_est[1],sigma_est[1])
y2 = mlab.normpdf(bins,mu_est[2],sigma_est[2])
plt.plot(bins,y0,'r--')
plt.plot(bins,y1,'r--')
plt.plot(bins,y2,'r--')
width = 0.7 * (bins[1]-bins[0])
center = (bins[:-1]+ bins[1:])/2
plt.bar(center,hist/(num_class*num_sample/num_bin),align='center',width=width)
plt.show()
