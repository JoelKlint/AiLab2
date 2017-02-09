import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,100)
y = x + 1
z = np.sin(x)
plt.plot(x, y)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid(True)
plt.show()