import numpy  as np
from scipy import linalg
np.random.seed(7)
d=10
a=np.random.choice([-1,1],d)
m_a=-np.ones(d)
beta=1.0
r=1-np.exp(-2.0*beta)
m=0
print(a)
## To allocate a number on a cluster.
for i in range(d):
    u=np.random.uniform()
    if(a[i]!=a[(i+d-1)%d] or u>r):
        m+=1
        m_a[i]=m
        max_m=m
    else:
        m_a[i]=m

## Treatment for the P.B.C.
for j in range(d):
    if(m_a[j]==0):
        m_a[j]=max_m
    else:
        break

print(m_a)
## Propose single cluster update.
sta=0
for m in range(1,max_m+1):
    temp=np.copy(a)
    flag=0
    for i in range(sta,d):
        if(m_a[i]==m):
            flag=1
            temp[i]*=-1
        elif(m_a[i]!=m and flag==1):
            sta=i
            break
    if(m==max_m):
        for j in range(d):
            if(m_a[j]==m):
                temp[j]*=-1
            else:
                break
    print("m=",m)
    print(temp)




