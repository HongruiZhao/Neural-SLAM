import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 100)

factor = 2
y_1 = np.exp(x)
y_2 = np.exp(factor*x)

plt.plot(x, y_1, label='exp(x)')
plt.plot(x, y_2, label=f'exp({factor}x)')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('exp.png')