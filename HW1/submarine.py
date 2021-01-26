#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.fft import fftn, ifftn, ifftshift
import matplotlib.pyplot as plt

subdata = np.load("subdata.npy", allow_pickle=True)

num_snapshots = subdata.shape[1]  # 49
L = 10  # length of space interval [-L, L]
n = int(subdata.shape[0]**(1/3) + 0.5)  # 64
s = np.linspace(-L, L, n, endpoint=False)
k_axes = ifftshift(2 * np.pi / (2 * L) * np.arange(-n // 2, n // 2))  # rescale wavenumbers bc fft assumes 2pi period

# ALGORITHM 1
# #--Averaging in frequency domain to reduce noise--#
X = []  # list of 64x64x64 arrays of spatial data in order from t=0:48
Xt = []  # frequency domain
for i in range(num_snapshots):
    x = np.reshape(subdata[:, i], (n, n, n))
    X.append(x)
    Xt.append(fftn(x))

xt_avg = sum(Xt) / num_snapshots


k_amax = np.unravel_index(np.argmax(abs(xt_avg)), xt_avg.shape)  # indices of the maximum 3D frequencies
k_max = tuple(k_axes[i] for i in k_amax)  # frequencies with max value
print("k_max:", k_max)

# ALGORITHM 2
# #--Gaussian Filter--#
scale = 3
A, B, C = np.meshgrid(k_axes, k_axes, k_axes)  # 3D fourier space. cubes with x-freqs, y-freqs, and z-freqs respectively
kernel = np.exp(-1 / scale**2 * ((A - k_max[0])**2 + (B - k_max[1])**2 + (C - k_max[2])**2))  #

coords = np.zeros((num_snapshots, 3))  # array of 49 3D coordinates of the submarine
for i in range(num_snapshots):
    denoised_xt = Xt[i] * kernel
    denoised_x = ifftn(denoised_xt)
    idxs = np.unravel_index(np.argmax(abs(denoised_x)), denoised_x.shape)
    coords[i, :] = s[idxs[0]], s[idxs[1]], s[idxs[2]]

# 3D trajectory
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], label="submarine path")
ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], label="start")
ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], label="end")
ax.view_init(elev=15, azim=50)
plt.xlabel("x")
plt.ylabel("y")
ax.set_zlabel("z")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
ax.set_xticks(np.linspace(-10, 10, 5, endpoint=True))
ax.set_yticks(np.linspace(-10, 10, 5, endpoint=True))
ax.set_zticks(np.linspace(-10, 10, 5, endpoint=True))
plt.legend()
plt.show()

# 2D projection
fig = plt.figure()
ax = fig.gca()
plt.plot(coords[:, 0], coords[:, 1], label="submarine path")
plt.scatter(coords[0, 0], coords[0, 1], label="start")
plt.scatter(coords[-1, 0], coords[-1, 1], label="end")
plt.xlabel("x")
plt.ylabel("y")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_xticks(np.linspace(-10, 10, 5, endpoint=True))
ax.set_yticks(np.linspace(-10, 10, 5, endpoint=True))
plt.legend()
plt.show()

# for table. use https://www.tablesgenerator.com/#
for j in range(3):
    for i in range(0, 49, 4):
        print(round(coords[i, j], 1), end=" ")
    print()

for i in range(0, 25, 2):
    print(f"{i}:00", end="\t")

print("\nfinal coords:", coords[-1, 0], coords[-1, 1], coords[-1, 2])
