import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz(x, y, z, s, r, b):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def attractor(x_0  , y_0, z_0 , dt , iteration , s , r , b):
    xs = np.empty(iteration + 1)
    ys = np.empty(iteration + 1)
    zs = np.empty(iteration + 1)
    xs[0], ys[0], zs[0] = (x_0, y_0, z_0)

    for i in range(iteration):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i],s,r,b)
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    return xs , ys , zs

dt = 0.001
iteration = 100000

xs , ys , zs = attractor(20 , 10 , 5 , dt , iteration , 10 , 28 , 2.666)

# print (attractor(20 , 10 , 5 , dt , iteration , 10 , 28 , 2.666))
# print (attractor(20 , 10 , 5 , dt , iteration , 10 , 28 , 2.666).size())


# Plot
# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# ax.plot(xs, ys, zs, lw=0.5)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("Lorenz Attractor")
# plt.show()