#!/usr/bin/env python
# coding: utf-8

# In[2]:


import scipy.io
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
get_ipython().run_line_magic('matplotlib', 'notebook')

from IPython.display import clear_output
import os

from scipy.ndimage import convolve


# In[3]:


def get_bw_avg(cams):
    camsbw = []  # black and white videos
    camavgs = []  # average frame in each video
    for i, cam in enumerate(cams):
        cambw = cam[:, :, 0, :] * 0.2989 + cam[:, :, 1, :] * 0.5870 + cam[:, :, 2, :] * 0.1140  # convert to grayscale
        camsbw.append(cambw)
        print("bw:", cambw.shape)
        camavg = np.mean(cambw, axis=2)
        camavgs.append(camavg)
    return camsbw, camavgs


# In[4]:


def preprocess(camsbw, camavgs, filts, kernel, step=1):
    filtereds = []
    for i, cambw in enumerate(camsbw):
        filtered = np.zeros((cambw.shape[0], cambw.shape[1], cambw.shape[2] // step))
        for t in range(0, filtered.shape[-1]):
            filtered[:, :, t] = convolve(cambw[:, :, t * step] - camavgs[i], kernel) * filts[i]
        filtereds.append(filtered)
    return filtereds


# In[5]:


def find_motion(filtereds, offset=2):
    camsdiff = []
    for i, filtered in enumerate(filtereds):
        camdiff = abs(filtered[:, :, offset:] - filtered[:, :, :-offset])
        camsdiff.append(camdiff)
    return filtereds


# In[6]:


def extract_location(camsdiff, dist_weight=0.5):
    tmax = camsdiff[0].shape[-1]
    grids = []
    for cam in camsdiff:
        grids.append(np.meshgrid(np.arange(cam.shape[1]), np.arange(cam.shape[0])))
    X = []
    i0, j0 = np.unravel_index(np.argmax(camsdiff[0][:,:,0]), camsdiff[0].shape[:2])
    i1, j1 = np.unravel_index(np.argmax(camsdiff[1][:,:,0]), camsdiff[1].shape[:2])
    i2, j2 = np.unravel_index(np.argmax(camsdiff[2][:,:,0]), camsdiff[2].shape[:2])
    X.append(np.array([i0, j0, i1, j1, i2, j2]))
    for t in range(1, tmax):
        i0, j0 = np.unravel_index(np.argmax(camsdiff[0][:,:,t] - dist_weight * ((grids[0][1] - X[-1][0])**2 + (grids[0][0] - X[-1][1])**2)), camsdiff[0].shape[:2])
        i1, j1 = np.unravel_index(np.argmax(camsdiff[1][:,:,t] - dist_weight * ((grids[1][1] - X[-1][2])**2 + (grids[1][0] - X[-1][3])**2)), camsdiff[1].shape[:2])
        i2, j2 = np.unravel_index(np.argmax(camsdiff[2][:,:,t] - dist_weight * ((grids[2][1] - X[-1][4])**2 + (grids[2][0] - X[-1][5])**2)), camsdiff[2].shape[:2])
        X.append(np.array([i0, j0, i1, j1, i2, j2]))
            
    return np.transpose(np.array(X, dtype=np.float32))


# In[7]:


def get_filter(xw, yw, scalex=400, scaley=400):
    filtX2, filtY2 = np.meshgrid(np.arange(xw), np.arange(yw), indexing="xy")
    filt = np.exp( -(filtX2 - xw/2)**2 / scalex**2 - (filtY2 - yw/2)**2 / scaley**2)
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.figure()
    plt.imshow(filt)
    plt.show()
    return filt


# In[8]:


width = 10
kernelX, kernelY = np.meshgrid(np.arange(-width, width), np.arange(-width, width))
scale = 6
kernel = np.exp( -(kernelX)**2 / scale**2 - (kernelY)**2 / scale**2)


# # Setup 1

# In[9]:


cam11 = scipy.io.loadmat("./cam1_1.mat")  # loads matlab files into python as a dict of np.ndarrays
cam11 = cam11["vidFrames1_1"].astype(np.float32)
cam21 = scipy.io.loadmat("./cam2_1.mat")["vidFrames2_1"].astype(np.float32)
cam31 = scipy.io.loadmat("./cam3_1.mat")["vidFrames3_1"].astype(np.float32)


# In[10]:


cams = [cam11[:, :480, :, 10:226], cam21[:, :480, :, 21:237], cam31[150:, :, :, 10:226]]


# In[11]:


camsbw, camavgs = get_bw_avg(cams)


# In[12]:


filt11 = get_filter(cams[0].shape[1], cams[0].shape[0], scalex=200, scaley=400)
filt21 = get_filter(cams[1].shape[1], cams[1].shape[0], scalex=200, scaley=400)
filt31 = get_filter(cams[2].shape[1], cams[2].shape[0], scalex=400, scaley=200)


# In[13]:


filtered = preprocess(camsbw, camavgs, [filt11, filt21, filt31], kernel, step=1)


# In[14]:


camsdiff = find_motion(filtered, offset=4)


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure()
ax = fig.gca()
plt.imshow(camavgs[0], cmap="gray")
plt.show()
for i in range(0, 15, 1):
    plt.imshow(camsdiff[0][:,:,i])
    plt.colorbar()
    plt.show()
    plt.imshow(camsbw[0][:,:,i], cmap="gray")
    plt.show()


# In[23]:


X = extract_location(camsdiff, dist_weight=0.5)


# In[24]:


plt.figure()
plt.plot(np.transpose(X)[:, :])
plt.show()


# In[25]:


X_c = X - np.mean(X, axis=1, keepdims=True)  # mean-subtracted
cov = X_c @ np.transpose(X_c) / (X_c.shape[-1] - 1)
L, V = np.linalg.eig(cov)
Y = np.transpose(V) @ X_c
covY = Y @ np.transpose(Y) / (Y.shape[-1] - 1)


# In[30]:


# covariance matrix visualization
plt.figure()
plt.imshow(np.log(abs(covY) + 1))
plt.colorbar()
plt.show()
covY


# In[27]:


# look at the relationship between the first two principle components
plt.scatter(Y[0], Y[1])
Y[0] @ Y[1] / (len(Y[0]) - 1)


# In[28]:


plt.figure()
plt.scatter(np.arange(len(L)), L)
# plt.semilogy()
plt.show()


# In[29]:


# calculate percent variance
plt.figure()
plt.plot(np.transpose(Y))
plt.show()


# # Setup 2 Camera Shake
# don't do any diffing frame to frame
# crop more
# 

# In[31]:


cam12 = scipy.io.loadmat("./cam1_2.mat")["vidFrames1_2"].astype(np.float32)
cam22 = scipy.io.loadmat("./cam2_2.mat")["vidFrames2_2"].astype(np.float32)
cam32 = scipy.io.loadmat("./cam3_2.mat")["vidFrames3_2"].astype(np.float32)


# In[32]:


print(cam12.shape, cam22.shape, cam32.shape)


# In[33]:


cams2 = [cam12[200:, 300:480, :, 4:314], cam22[:, 150:480, :, 26:336], cam32[150:, :, :, 7:317]]
camsbw, camavgs = get_bw_avg(cams2)


# In[34]:


filt12 = get_filter(cams2[0].shape[1], cams2[0].shape[0], scalex=200, scaley=400)
filt22 = get_filter(cams2[1].shape[1], cams2[1].shape[0], scalex=200, scaley=400)
filt32 = get_filter(cams2[2].shape[1], cams2[2].shape[0], scalex=400, scaley=200)


# In[35]:


filts = [filt12, filt22, filt32]
step=2
filtered = []
for i, cam in enumerate(cams2):
    fltd = np.zeros((cam.shape[0], cam.shape[1], cam.shape[-1] // step))
    for t in range(0, fltd.shape[-1]):
        frame = camsbw[i][:, :, t*step]
        deviation = 2 * abs(cam[:, :, 0, t*step] - frame) + 3 * abs(cam[:, :, 0, t*step] - frame) + abs(cam[:, :, 0, t*step] - frame)
        fltd[:, :, t] = (frame - 0.5*deviation) * filts[i]
    filtered.append(fltd)


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure()
ax = fig.gca()
plt.imshow(camavgs[1], cmap="gray")
plt.show()
for i in range(100, 155, 1):
    plt.imshow(filtered[1][:,:,i])
    plt.colorbar()
    plt.show()
    plt.imshow(camsbw[1][:,:,i], cmap="gray")
    plt.show()


# In[37]:


X2 = extract_location(filtered, dist_weight=0.02)


# In[38]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.figure()
plt.plot(np.transpose(X2)[:, :])
plt.show()


# In[39]:


X2_c = X2 - np.mean(X2, axis=1, keepdims=True)  # mean-subtracted
cov = X2_c @ np.transpose(X2_c) / (X2_c.shape[-1] - 1)
L2, V = np.linalg.eig(cov)
Y2 = np.transpose(V) @ X2_c
covY = Y2 @ np.transpose(Y2) / (Y2.shape[-1] - 1)


# In[40]:


L2


# In[41]:


plt.figure()
plt.scatter(np.arange(len(L2)), L2)
# plt.semilogy()
plt.xlabel("index")
plt.ylabel("$\sigma$")
plt.show()


# In[43]:


plt.figure()
plt.plot(np.transpose(Y2)[:, :3])
plt.show()


# In[ ]:





# # Setup 3 Oscillation and Pendulum

# In[44]:


cam13 = scipy.io.loadmat("./cam1_3.mat")["vidFrames1_3"].astype(np.float32)
cam23 = scipy.io.loadmat("./cam2_3.mat")["vidFrames2_3"].astype(np.float32)
cam33 = scipy.io.loadmat("./cam3_3.mat")["vidFrames3_3"].astype(np.float32)


# In[45]:


cam33.shape


# In[46]:


cams3 = [cam13[:, :450, :, 2:219], cam23[:, :450, :, 32:249], cam33[200:, :, :, 20:]]
camsbw, camavgs = get_bw_avg(cams3)


# In[47]:


filt13 = get_filter(cams3[0].shape[1], cams3[0].shape[0], scalex=400, scaley=400)
filt23 = get_filter(cams3[1].shape[1], cams3[1].shape[0], scalex=400, scaley=400)
filt33 = get_filter(cams3[2].shape[1], cams3[2].shape[0], scalex=400, scaley=400)


# In[ ]:





# In[48]:


filtered = preprocess(camsbw, camavgs, [filt13, filt23, filt33], kernel, step=2)


# In[49]:


camsdiff = find_motion(filtered, offset=4)


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure()
ax = fig.gca()
plt.imshow(camavgs[0], cmap="gray")
plt.show()
for i in range(0, 10, 1):
#     plt.imshow(np.log(abs(camsbw[0][:,:,i + 1] - camsbw[0][:,:,i]) + 1))
    plt.imshow(camsdiff[0][:,:,i])
    plt.colorbar()
    plt.show()
    plt.imshow(camsbw[0][:,:,i], cmap="gray")
    plt.show()


# In[51]:


X3 = extract_location(camsdiff, dist_weight=0.5)


# In[52]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.figure()
plt.plot(np.transpose(X3)[:, :])
plt.show()


# In[53]:


X3_c = X3 - np.mean(X3, axis=1, keepdims=True)  # mean-subtracted
cov = X3_c @ np.transpose(X3_c) / (X3_c.shape[-1] - 1)
L3, V = np.linalg.eig(cov)
Y3 = np.transpose(V) @ X3_c
covY = Y3 @ np.transpose(Y3) / (Y3.shape[-1] - 1)


# In[54]:


plt.figure()
plt.scatter(np.arange(len(L3)), L3)
# plt.semilogy()
plt.show()


# In[56]:


plt.figure()
plt.plot(np.transpose(Y3)[:, [0, 1, 3]])
plt.show()


# In[ ]:





# # Setup 4 Oscillation and Rotation

# In[57]:


cam14 = scipy.io.loadmat("./cam1_4.mat")["vidFrames1_4"].astype(np.float32)
cam24 = scipy.io.loadmat("./cam2_4.mat")["vidFrames2_4"].astype(np.float32)
cam34 = scipy.io.loadmat("./cam3_4.mat")["vidFrames3_4"].astype(np.float32)


# In[58]:


cams4 = [cam14[:, :460, :, :192], cam24[:, :460, :, 4:196], cam34[100:, 300:, :, :192]]
camsbw, camavgs = get_bw_avg(cams4)


# In[59]:


filt14 = get_filter(cams4[0].shape[1], cams4[0].shape[0], scalex=400, scaley=400)
filt24 = get_filter(cams4[1].shape[1], cams4[1].shape[0], scalex=400, scaley=400)
filt34 = get_filter(cams4[2].shape[1], cams4[2].shape[0], scalex=400, scaley=400)


# In[60]:


filtered = preprocess(camsbw, camavgs, [filt14, filt24, filt34], kernel, step=2)


# In[61]:


camsdiff = find_motion(filtered, offset=4)


# In[62]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure()
ax = fig.gca()
plt.imshow(camavgs[2], cmap="gray")
plt.show()
for i in range(0, 30, 1):
    plt.imshow(camsdiff[2][:,:,i])
    plt.colorbar()
    plt.show()
    plt.imshow(camsbw[2][:,:,i], cmap="gray")
    plt.show()


# In[63]:


X4 = extract_location(camsdiff, dist_weight=0.5)


# In[64]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.figure()
plt.plot(np.transpose(X4)[:, :])
plt.show()


# In[65]:


X4_c = X4 - np.mean(X4, axis=1, keepdims=True)  # mean-subtracted
cov = X4_c @ np.transpose(X4_c) / (X4_c.shape[-1] - 1)
L4, V = np.linalg.eig(cov)
Y4 = np.transpose(V) @ X4_c
covY = Y4 @ np.transpose(Y4) / (Y4.shape[-1] - 1)


# In[66]:


plt.figure()
plt.scatter(np.arange(len(L4)), L4)
# plt.semilogy()
plt.show()


# In[68]:


plt.figure()
plt.plot(np.transpose(Y4)[:, :3])
plt.show()


# In[ ]:





# # Figures

# In[318]:


get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams['figure.dpi'] = 100

width = 10
height = 6

fig, axs = plt.subplots(2, 2, figsize=(width, height))

plt.subplot(2, 2, 1)
plt.plot(np.transpose(X))
plt.xlabel("t (frames)", fontsize=13)
plt.ylabel("position", fontsize=13)
plt.title("case 1 recordings")

plt.subplot(2, 2, 2)
plt.plot(np.transpose(X2))
plt.xlabel("t (frames)", fontsize=13)
plt.ylabel("position", fontsize=13)
plt.title("case 2 recordings")

plt.subplot(2, 2, 3)
plt.plot(np.transpose(X3))
plt.xlabel("t (frames)", fontsize=13)
plt.ylabel("position", fontsize=13)
plt.title("case 3 recordings")

plt.subplot(2, 2, 4)
plt.plot(np.transpose(X4))
plt.xlabel("t (frames)", fontsize=13)
plt.ylabel("position", fontsize=13)
plt.title("case 4 recordings")

# plt.legend(["cam 1 y", "cam 1 x", "cam 2 y", "cam 2 x", "cam 3 y", "cam 3 x"], bbox_to_anchor=(0, -0.5), loc="upper right")
plt.tight_layout()
plt.show()


# In[333]:


width = 10
height = 10

fig, axs = plt.subplots(4, 2, figsize=(width, height))

plt.subplot(4, 2, 1)
plt.scatter(np.arange(len(L)), np.sort(L)[::-1] / (len(L) - 1))
plt.xlabel("index", fontsize=13)
plt.ylabel("$\sigma^2$", fontsize=13)
plt.title("case 1 explained variance")

plt.subplot(4, 2, 2)
plt.plot(np.transpose(Y[0]))
plt.xlabel("t (frames)", fontsize=13)
plt.ylabel("projected position", fontsize=13)
plt.title("case 1 principle component projection")



plt.subplot(4, 2, 3)
plt.scatter(np.arange(len(L2)), np.sort(L2)[::-1] / (len(L2) - 1))
plt.xlabel("index", fontsize=13)
plt.ylabel("$\sigma^2$", fontsize=13)
plt.title("case 2 explained variance")

plt.subplot(4, 2, 4)
plt.plot(np.transpose(Y2[[0, 1, 2]]))
plt.xlabel("t (frames)", fontsize=13)
plt.ylabel("projected position", fontsize=13)
plt.title("case 2 principle component projection")



plt.subplot(4, 2, 5)
plt.scatter(np.arange(len(L3)), np.sort(L3)[::-1] / (len(L3) - 1))
plt.xlabel("index", fontsize=13)
plt.ylabel("$\sigma^2$", fontsize=13)
plt.title("case 3 explained variance")

plt.subplot(4, 2, 6)
plt.plot(np.transpose(Y3[[0, 1]]))
plt.xlabel("t (frames)", fontsize=13)
plt.ylabel("projected position", fontsize=13)
plt.title("case 3 principle component projection")



plt.subplot(4, 2, 7)
plt.scatter(np.arange(len(L4)), np.sort(L4)[::-1] / (len(L4) - 1))
plt.xlabel("index", fontsize=13)
plt.ylabel("$\sigma^2$", fontsize=13)
plt.title("case 4 explained variance")

plt.subplot(4, 2, 8)
plt.plot(np.transpose(Y4[[0, 1, 2]]))
plt.xlabel("t (frames)", fontsize=13)
plt.ylabel("projected position", fontsize=13)
plt.title("case 4 principle component projection")


plt.tight_layout()
plt.show()


# In[69]:


sum(L[:1]) / sum(L)


# In[325]:


sum(L2[:3]) / sum(L2)


# In[328]:


sum(L3[:2]) / sum(L3)


# In[327]:


sum(L4[:3]) / sum(L4)


# In[ ]:




