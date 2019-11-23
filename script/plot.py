__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/23 02:48:08"

import numpy as np
import matplotlib as mpl
import pickle
import matplotlib.pyplot as plt

X = []
for index in range(1,5):
    with open(f"./output/learn_by_potential_x_transformed_from_z_{index}.pkl", 'rb') as file_handle:
        data = pickle.load(file_handle)
        X.append(data['x'])

fig = plt.figure(0, figsize = (12.8, 4.8))
fig.clf()
for index in range(1, 5):
    ax = plt.subplot(1,4,index)
    plt.plot(X[index-1][:,0], X[index-1][:,1], '.', alpha = 0.5)
    plt.xlim((-4, 4))
    plt.ylim((-4, 4))
    ax.set_aspect(1.0)
plt.savefig("./output/learn_by_potential_x_transformed_from_z.png")


        
