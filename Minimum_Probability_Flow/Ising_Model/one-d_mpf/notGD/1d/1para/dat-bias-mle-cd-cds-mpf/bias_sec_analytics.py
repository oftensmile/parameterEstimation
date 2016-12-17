import numpy as np
d=16
def log_z(d,J):
    sh=np.sinh(J)
    ch=np.cosh(J)
    return (d*sh*ch*(ch**(d-2)+sh**(d-2)))/(sh**d+ch**d)

if __name__ == '__main__':
    dJ=0.00001
    """
    J_min,J_max=0.9999,1.0001
    N=int((J_max-J_min)/dJ)
    for n in range(N):
    #for J in J_list:
        J=J_min+dJ*n
        lz=log_z(d,J)
        dlz=(log_z(d,J+dJ)-log_z(d,J-dJ))/(2.0*dJ)
        ddlz=(log_z(d,J+dJ)-2*lz+log_z(d,J-dJ))/dJ**2
        print(J,lz,dlz,ddlz)  
    """
    J=1.000
    lz=log_z(d,J) #= (-1) * expectation of energy
    dlz=(log_z(d,J+dJ)-log_z(d,J-dJ))/(2.0*dJ) #=specific heat,C(J0)
    ddlz=(log_z(d,J+dJ)-2*lz+log_z(d,J-dJ))/dJ**2 #=derivative of the specific heat,C'(J0)
    #print(J,lz,dlz,ddlz) 

    bias=0.0 #J0=1.0; =C(J0)**(-2)C'(J0)/N
    for n in range(1,1000):
        print(n,dlz**(-2)*ddlz/n)
