import numpy as np
from scipy import linalg 
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d.axes3d import Axes3D
np.set_printoptions(precision=3)
np.random.seed(0)
d,T,N,heatN,J=4,1000,1000,20,1
h,lam=1.0**2,0.1
theta=[[1 if i==(j+1+d)%d or i==(j-1+d)%d else 0 for i in range(d)] for j in range(d)]
def gen_mcmc(t_wait, x=[],theta=[[]]):
    for t in range(t_wait):
        for i in range(d):
            valu=J*(np.dot(theta[:][i],x)-x[i]*theta[i][i])
            r=np.exp(-valu)/(np.exp(-valu)+np.exp(valu))
            R=np.random.uniform(0,1)
            if(R<=r):
                x[i]=1
            else:
                x[i]=-1
    return x

def sum_xixj(n_sample,theta=[[]]):
    xixj=np.zeros((d,d))
    y=np.ones(d)
    y=gen_mcmc(500,y,theta)
    for n in range(n_sample):
        y=gen_mcmc(3,y,theta)
        xixj=xixj+np.tensordot(y,y,axes=([0][0]))
    for l in range(d):xixj[l][l]=0
    return xixj/n_sample

#sampling of the theta matrix
def gen_theta(t_wait,samp_theta=[[]],X=[[]]):
    for t in range(t_wait):
        for i in range(d):
            for j in range(i+1,d):
                dtheta=0.1*np.random.uniform(-1,1)#theta(t)-theta(t+1)
                value=J*dtheta*X[i][j]
                if(value>0):
                    r=np.exp(-value)
                    R=np.random.uniform(0,1)
                    if(r<R):
                        samp_theta[i][j]=samp_theta[i][j]-dtheta
                        samp_theta[j][i]=samp_theta[j][i]-dtheta
    return samp_theta


def kernel_regres(x,target):
    xx=np.dot(np.matrix(x),np.matrix(x).T)
    x2=np.diagonal(xx)
    x2=np.matlib.repmat(np.matrix(x2).T,1,len(x))+np.matlib.repmat(np.matrix(x2),len(x),1)
    K=np.exp(-(x2-2.0*xx)/h)
    Kinv=np.linalg.inv(K+lam*np.diag(np.ones(len(x))))#+lam*np.diag(np.ones(len(x)))
    return np.dot(Kinv,target)

def k_of_x(n_sample,x,sample):
    kofx=np.zeros(n_sample)
    for i in range(n_sample):
        kofx[i]=np.exp(-(np.dot(sample[:,i].T,sample[:,i])+np.dot(x,x)-2*np.dot(x,sample[:,i]))/h)
    return kofx
#################### MAIN #########################
dl_sample=sum_xixj(N,theta)
# initial theta
theta_est=0.1*np.random.rand(d,d)
theta_est=0.5*(theta_est + theta_est.T)
loss,delta=np.zeros(T),np.zeros(T)
loss_mcmc=np.zeros(T)
samp_theta=np.ones((d,d),dtype=np.float)
r,a=0.0,2.0
for t in range(T):
    #SGD-method
    dl_model=sum_xixj(heatN,theta_est)
    r+=np.sum(dl_model*dl_model)
    lr=a/(np.sqrt(r)+epc)
    grad=dl_sample-dl_model
    theta_est=theta_est-lr*grad
    loss[t]=np.absolute(theta-theta_est).sum()
    delta[t]=np.absolute(grad).sum()
    #symple-MCMC-method
    samp_theta=gen_theta(3,samp_theta,dl_sample)
    loss_mcmc[t]=np.absolute(theta-samp_theta).sum()
     
    #samp_xixj=sum_xixj(100,samp_theta)
    #samp_theta_vec=np.reshape(samp_theta,(1,d**2))#[t[0,0],t[0,1],..,t[0,d-1],t[1,0],..
    #if t==0:
    #    stack_theta=samp_theta_vec
    #    stack_x1x2=samp_xixj[1][2]
    #else:
    #    stack_theta=np.vstack((stack_theta,samp_theta_vec))
    #    stack_x1x2=np.vstack((stack_x1x2,samp_xixj[1][2]))
step=np.arange(T)
sgd,=plt.plot(step,loss,'r-',label='SGD')
mcs,=plt.plot(step,loss_mcmc,'b-',label='mcmc')
plt.legend([sgd,mcs],['SGD','mcmc'])
plt.show()

#visualization of naive sample data, 1 to 3
#fig=plt.figure()
#ax=Axes3D(fig)
#ax.scatter3D(np.ravel(stack_theta[:,1]),np.ravel(stack_theta[:,2]),np.ravel(stack_theta[:,3]),c=np.array(stack_x1x2))
#plt.show()

#H=np.matrix(np.eye(d**2)-np.ones((d**2,d**2)))
#cov_theta=np.matrix(np.dot(stack_theta.T,stack_theta))
#cov_theta=np.dot(np.dot(H,cov_theta),H)
#cov_theta=np.dot(stack_theta.T,stack_theta)
#print(np.shape(stack_theta[:,1])) = (5,)
#alpha=kernel_regres(stack_theta,stack_x1x2)
#la,v=linalg.eig(cov_theta)
#idx=la.argsort()[::-1]
#l1,l2,l3=la[idx[0]],la[idx[1]],la[idx[2]]
#v1,v2,v3=v[:,idx[0]],v[:,idx[1]],v[:,idx[2]]

#Tpca=np.vstack((np.vstack((v1,v2)),v3))
#projected=np.dot(np.matrix(Tpca),np.matrix(stack_theta).T)
#base1,base2,base3=np.real(projected[0]),np.real(projected[1]),np.real(projected[2])   
#fig=plt.figure()
#ax=Axes3D(fig)
#ax.scatter3D(np.ravel(base1),np.ravel(base2),np.ravel(base3),c=np.real(stack_x1x2))
#p=ax.scatter(np.ravel(base1),np.ravel(base2),np.ravel(base3),c=np.real(stack_x1x2))
#fig.colorbar(p)
#plt.show()



#max1,min1=np.max(base1),np.min(base1)
#max2,min2=np.max(base2),np.min(base2)
#Nmesh=100
#axis1=np.linspace(min1,max1,Nmesh)
#axis2=np.linspace(min2,max2,Nmesh)
#Z=np.zeros((Nmesh,Nmesh))
#for l1 in range(Nmesh):
#    for l2 in range(Nmesh):
#        Z[l1][l2]=np.dot(alpha.T,k_of_x(n_sample,[axis1[l1],axis2[l2]],pbase))

#print("np.shape(Tpca_covtheta)",np.shape(pbase))
#print("shape Z[0][0]",np.shape(Z[0][0]))
#print("Z shape =",np.shape(Z))
#print("Z[0][0]=",Z[0][0])
#fig=plt.figure()
#ax=Axes3D(fig)
#ax.plot_wireframe(axis1,axis2,Z)
#ax.scatter3D(np.ravel(base1),np.ravel(base2),np.ravel(stack_x1x2))

#alpha=kernel_regres(stack_theta,stack_x1x2)
#chack t[1,2]=t[d+2-1]element only 
#tmax,tmin=np.max(stack_x1x2),np.min(stack_x1x2)
#theta_line=np.linspace(tmin,tmax,100)
#for l in range(len(theta_line)):
#    if l==0:
#        k=k_of_x(theta_line[l],stack_theta[:,d+2-1])
#    else:
#        k=np.vstack((k,k_of_x(theta_line[l],stack_theta[:,d+2-1])))
#
#Y=np.dot(np.matrix(k),np.matrix(alpha))
#
#p1,=plt.plot(stack_theta[:,(d+2-1)%(d**2)],stack_x1x2,'o')
#p2,=plt.plot(theta_line,Y,'-')
#plt.legend([p1,p2],['theta(1,2)','target(theta)'])
#plt.show()






#plt.subplot(131)
#plt.imshow(dl_sample)
#plt.colorbar()
#plt.title('gram_matrix')
#plt.subplot(132)
#plt.imshow(samp_theta)
#plt.colorbar()
#plt.title('samp_theta')
#plt.subplot(133)
#plt.imshow(phi)
#plt.colorbar()
#plt.title('gram_matrix(samp_theta)')
#plt.show()


#for l in range(d):theta_est[l][l]=0
#using AdaGrad
#r,a,epc=0,2,0.001
#for k in range(T):
#    dl_model=sum_xixj(heatN,theta_est)
#    r+=np.sum(dl_model*dl_model)
#    lr=a/(np.sqrt(r)+epc)
#    theta_est=theta_est-lr*(dl_sample - dl_model)
#    loss[k]=np.absolute(theta-theta_est).sum()

#result=[gen_mcmc(1000,np.ones(d),theta_est)]
#plt.subplot(221)
#plt.imshow(theta)
#plt.colorbar()
#plt.title('true theta')
#plt.subplot(222)
#plt.imshow(theta_est)
#plt.colorbar()
#plt.title('estimated theta ')
#plt.subplot(223)
#plt.imshow(result)
#plt.title('generated config')
#plt.subplot(224)
#plt.plot(loss)
#plt.title('loss function')
#plt.show()


