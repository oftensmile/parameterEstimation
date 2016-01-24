import numpy as np
np.random.seed(0)
def derivative(f):
    def compute(x, dx):
        return (f(x+dx) - f(x))/dx
    return compute

def newtons_method(f, x, dx=0.000001, tolerance=0.000001):
    df = derivative(f)
    while True:
        x1 = x - f(x)/df(x, dx)
        for i in range(len(x)):
            t = abs(x1[i] - x[i])
            if (t < tolerance):
                break
            x[i] = x1[i]
    return x
# 一先ず、vectorで上手くいくか試してみる
def f(x):
    return 3*x**5 - 2*x**3 + 1*x - 37

x_approx = np.arange(-3,3)  # rough guess
# f refers to the function f(x)
x = newtons_method(f, x_approx)

print("Solve for x in 3*x**5 - 2*x**3 + 1*x - 37 = 0")
print("x = %0.12f" % x)

''' result ...
Solve for x in 3*x**5 - 2*x**3 + 1*x - 37 = 0
x = 1.722575335786
'''
# optional test (result should be close to zero)
# change dx and tolerance level to make it a little closer
print("Testing with the above x value ...")
print("%0.12f" % (3*x**5 - 2*x**3 + 1*x - 37))  
''' result ...
0.000000399251
'''
