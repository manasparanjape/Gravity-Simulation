import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
import glob

start = datetime.now()
n = 100  # Number of particles
G = 1000  # Gravitational constant
particle_size = 1  # Particle size, so that 2 particles cannot come closer than this dist
time_step = 0.1
total_time = 36
total_steps = int(total_time/time_step)

x_lim = 1000
y_lim = 1000

M = 2 * np.random.uniform(low=0.1, high=1, size=(n,))
s = [50*M[n] for n in range(len(M))]
X = x_lim * np.random.rand(n)
Y = y_lim * np.random.rand(n)

F = np.zeros((n, n))
acc_x = np.zeros((n, n))
acc_y = np.zeros((n, n))
theta = np.zeros((n, n))
ux = np.zeros((n, n))
uy = np.zeros((n, n))
sx = np.zeros((n, n))
sy = np.zeros((n, n))
vx = np.zeros((n, n))
vy = np.zeros((n, n))
Xt = np.zeros((n, total_steps))
Yt = np.zeros((n, total_steps))
current_pos = np.zeros((n, 2))

current_pos[:, 0] = X
current_pos[:, 1] = Y

print('\n### Boundary Conditions ###')
print('Number of particles:', n)
print('Particle size:', particle_size)
print('Gravitational constant:', G)
print('End time:', total_time)
print('Time step:', time_step)
print('Total steps:', total_steps)

print('\nRunning simulation...')
t = 0
status = 0

for z in range(total_steps):
    status = status + 1
    p = np.round((status/total_steps)*100, 2)
    print('\r''Progress:', p, '%', end='')
    for i in range(n):
        for j in range(n):
            if i != j:
                r = (X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2
                r = max(r, (2 * particle_size))
                F[i, j] = (G * M[i] * M[j]) / r

                if X[i] - X[j] == 0:
                    if (Y[i] - Y[j]) < 0:
                        theta[i, j] = np.pi/2
                    else:
                        theta[i, j] = (3*np.pi)/2
                elif Y[i] - Y[j] == 0:
                    if (X[i] - X[j]) < 0:
                        theta[i, j] = 0
                    else:
                        theta[i, j] = np.pi
                elif Y[i] - Y[j] == 0 and X[i] - X[j] == 0:
                    theta = 0
                else:
                    theta[i, j] = np.arctan(abs(Y[i] - Y[j]) / abs(X[i] - X[j]))
                    if (X[i] - X[j]) < 0:
                        if (Y[i] - Y[j]) < 0:
                            theta[i, j] = theta[i, j]  # First Quadrant wrt i
                        else:
                            theta[i, j] = (2*np.pi) - theta[i, j]  # Fourth Quadrant wrt i
                    else:
                        if (Y[i] - Y[j]) < 0:
                            theta[i, j] = np.pi - theta[i, j]
                        else:
                            theta[i, j] = np.pi + theta[i, j]

    Fx = np.multiply(F, np.cos(theta))
    Fy = np.multiply(F, np.sin(theta))

    for i in range(n):
        for j in range(n):
            acc_x[i, j] = Fx[i, j]/M[i]
            acc_y[i, j] = Fy[i, j] / M[i]

    for i in range(n):
        for j in range(n):
            sx[i, j] = (ux[i, j] * time_step) + (0.5 * acc_x[i, j] * time_step * time_step)
            sy[i, j] = (uy[i, j] * time_step) + (0.5 * acc_y[i, j] * time_step * time_step)
            vx[i, j] = ux[i, j] + (acc_x[i, j] * time_step)
            vy[i, j] = uy[i, j] + (acc_y[i, j] * time_step)
            if (X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2 <= 2:
                if i <= j:
                    tempx = vx[i]
                    tempy = vy[i]
                    vx[i] = ((M[i] - M[j]) / (M[i] + M[j])) * vx[i] + (2 * M[j] / (M[i] + M[j])) * vx[j]
                    vy[i] = ((M[i] - M[j]) / (M[i] + M[j])) * vy[i] + (2 * M[j] / (M[i] + M[j])) * vy[j]
                    vx[j] = ((M[j] - M[i]) / (M[j] + M[i])) * vx[j] + (2 * M[i] / (M[j] + M[i])) * tempx
                    vy[j] = ((M[j] - M[i]) / (M[j] + M[i])) * vy[j] + (2 * M[i] / (M[j] + M[i])) * tempy

    ux = vx.copy()
    uy = vy.copy()

    for i in range(n):
        X[i] = X[i]+np.sum(sx[i, :])
        Y[i] = Y[i] + np.sum(sy[i, :])
        if X[i] > x_lim:
            X[i] = X[i] - x_lim
        if X[i] < 0:
            X[i] = x_lim - X[i]
        if Y[i] > y_lim:
            Y[i] = Y[i] - y_lim
        if Y[i] < 0:
            Y[i] = y_lim - Y[i]

    Xt[:, z] = X
    Yt[:, z] = Y

    t = t + time_step

    plt.scatter(X, Y, s=s, color='black')
    plt.xlim([0, 1000])
    plt.ylim([0, 1000])
    plt.axis('off')
    filename = 'frame' + str(status) + '.png'
    # sys.stdout.flush()
    plt.savefig(filename)
    plt.close()


