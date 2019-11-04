__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/03 20:25:17"

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

N = 2000
x1 = np.linspace(-1, 1, N)
y1 = -2*x1**2 + 1 + np.random.normal(scale = 0.15, size = (N,))

N = N//2
x2 = np.linspace(0, 2, N)
y2 = 2*(x2 - 1)**2 - 2 + np.random.normal(scale = 0.15, size = (N,))

data = np.vstack((np.concatenate((x1, x2)), np.concatenate((y1,y2)))).T

fig = plt.figure(0)
fig.clf()
plt.plot(data[:,0], data[:,1], ".")
plt.savefig("./output/tmp.pdf")

with open("./output/data.pkl", 'wb') as file_handle:
    pickle.dump(data, file_handle)
    
