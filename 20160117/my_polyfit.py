import numpy as np
import matplotlib.pyplot as plt
x=np.arange(0,2*np.pi,0.2)
y=np.array(np.sin(x))+ np.random.random(len(x))
z=np.polyfit(x,y,5)
p=np.poly1d(z)
p30=np.poly1d(np.polyfit(x,y,33))
xp=np.linspace(-2,6,100)

plt.plot(x,y,'.',xp,p(xp),'-',xp,p30(xp),'*')
plt.ylim(-2,2)
plt.show()
plt.savefig('image2.png')

