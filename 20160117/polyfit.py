import numpy as np
import matplotlib.pyplot as plt
x=np.arange(6)
y=np.array([0.0,0.8,0.9,0.1,-0.8,-1.0])
z=np.polyfit(x,y,3)
p= np.poly1d(z)
p30=np.poly1d(np.polyfit(x,y,30))
xp=np.linspace(-2,6,100)

plt.plot(x,y,'.',xp,p(xp),'-',xp,p30(xp),'*')
plt.ylim(-2,2)
plt.show()
plt.saavefig('image.png')
