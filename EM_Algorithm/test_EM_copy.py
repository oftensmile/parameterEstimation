import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def draw_hist(xs,bins):
    plt.hist(xs,bins=bins,normed=True,alpha=0.5)

def predict(data):
    mu=np.mean(data)
    var=np.var(data,ddof=1)
    return (mu,var)
def main():
    mu,v=3.0,2.0
    std=np.sqrt(v)
    N=10000
    data=np.random.normal(mu,std,N)
    mu_predicted, var_predicted=predict(data)
    std_predicted=np.sqrt(var_predicted)
    print('original: mu={0}, var={1}'.format(mu,v))
    print(' predict: mu={0}, var={1}'.format(mu_predicted,var_predicted))

    #draw our resolut
    draw_hist(data, bins=40)
    xs=np.linspace(min(data),max(data),200)
    norm=mlab.normpdf(xs, mu_predicted, std_predicted)
    plt.plot(xs,norm,color='red')
    plt.xlim(min(xs),max(xs))
    plt.xlabel('x')
    plt.ylabel('probability')
    plt.show()

if __name__ =='__main__':
    main()
