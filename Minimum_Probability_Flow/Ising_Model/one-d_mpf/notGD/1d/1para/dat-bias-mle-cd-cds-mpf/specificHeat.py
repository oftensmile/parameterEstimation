import numpy as np
d=16
def log_z(d,J):
    sh=np.sinh(J)
    ch=np.cosh(J)
    return (d*sh*ch*(ch**(d-2)+sh**(d-2)))/(sh**d+ch**d)
    #return np.log((2.0*np.cosh(J))**d+(2.0*np.sinh(J))**d)
    #return np.sin(J) 
if __name__ == '__main__':
    dJ=0.00001
    #dJ=0.001*3.14
    #J_min,J_max=0.95,7.05
    J_min,J_max=0.9999,1.0001
    
    N=int((J_max-J_min)/dJ)
    for n in range(N):
    #for J in J_list:
        J=J_min+dJ*n
        lz=log_z(d,J)
        dlz=(log_z(d,J+dJ)-log_z(d,J-dJ))/(2.0*dJ)
        ddlz=(log_z(d,J+dJ)-2*lz+log_z(d,J-dJ))/dJ**2
        print(J,lz,dlz,ddlz)  

