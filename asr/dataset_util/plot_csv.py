import numpy as np
from matplotlib import pyplot as plt


CSV_FILE = '/home/marc/Downloads/test.csv'

data = np.loadtxt(CSV_FILE, skiprows=1, delimiter=',')

data = data[:, 1:]
print('data:', data)

plt.figure()
plt.plot(data[:, [0]], data[:, [1]], linestyle='-', linewidth=0.9, color='#1111EE', marker='o')
plt.grid(visible=True)
plt.ylabel('WER')
plt.xlabel('Step')
plt.title('TITLE', visible=False)
plt.show()
