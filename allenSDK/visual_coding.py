#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import shutil

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
font = {'size': 16}
matplotlib.rc('font', **font)

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
# In[2]:
data_directory = './ecephys_cache' # must be updated to a valid directory in your filesystem
manifest_path = os.path.join(data_directory, "manifest.json")
# In[3]:
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
# In[4]:
sessions = cache.get_session_table()
print('Total number of sessions: ' + str(len(sessions)))
# In[5]:
filtered_sessions = sessions[(sessions.full_genotype.str.find('Sst') > -1) & (sessions.session_type == 'brain_observatory_1.1') & (['VISp' in acronyms for acronyms in sessions.ecephys_structure_acronyms])]
print(len(filtered_sessions))
filtered_sessions.head()
# In[6]:
session_id = 715093703  # based on the above filter
session = cache.get_session_data(session_id)
# In[7]:
units = cache.get_units()
# In[9]:
unit_ids = units[(units.ecephys_structure_acronym=="VISp") & (units.specimen_id==filtered_sessions.specimen_id[session_id])].index
unit_ids.shape

# # Static Grating Classification
# ## Controlled orientation decoding

# In[27]:
stim_epochs = session.get_stimulus_epochs()
stim_epochs = stim_epochs[stim_epochs.stimulus_name=="static_gratings"]
stim_epochs
# In[28]:
stims = session.get_stimulus_table(['static_gratings'])
stims = stims[stims.orientation!='null']
# In[19]:
filtered_stims = stims[(stims.phase=="0.0") & (stims.spatial_frequency=="0.04")]
filtered_stims
# In[20]:
# In[21]:
num_bins = 10
stim_duration = filtered_stims.duration.min()  # minimum so that none of the recording periods are overlapping
bin_edges = np.linspace(0, stim_duration, num_bins + 1)
counts = session.presentationwise_spike_counts(bin_edges=bin_edges,
                                               stimulus_presentation_ids=filtered_stims.index,
                                               unit_ids=unit_ids)
# In[22]:
oris = np.unique(filtered_stims.orientation).astype(int)
oris
# In[23]:
avg_resp = np.mean(counts, axis=0)
# In[26]:
fig, ax = plt.subplots(2, 3, figsize=(10, 8), dpi=60)
fig.suptitle("mean-subtracted response to static gratings")
for i, ori in enumerate(oris):
    plt.subplot(2, 3, i+1)
    idxs = filtered_stims[filtered_stims.orientation==ori].index
    plt.ylabel("unit")
    plt.xlabel("time (ms)")
#     im = plt.imshow((np.mean(counts.loc[idxs], axis=0).T - avg_resp), aspect=0.2, vmin=-0.5, vmax=1.1)
    im = plt.imshow(counts.loc[idxs[0]].T - avg_resp, aspect=0.2, vmin=-0.5, vmax=1.1)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, num_bins, 2))
    ax.set_xticklabels((2000 * np.arange(num_bins // 2) * stim_duration / num_bins).astype(int))
    plt.title(f"{ori}$^\circ$ grating")
cbar_ax = fig.add_axes([1.0, 0.15, 0.05, 0.7])
fig.colorbar(mappable=im, label=f"mean-subtracted number of spikes per {int(1000 * stim_duration / num_bins)} ms", cax=cbar_ax)
plt.tight_layout()
plt.show()

# ## Data preprocessing and PCA
# In[522]:
np.random.seed(10)
# In[523]:
Xall = np.reshape(np.array(counts), (counts.shape[0], num_bins * len(unit_ids)))
labels = np.array(list(filtered_stims.orientation))
# train test split
train_frac = 0.7
train_idxs = np.array([], dtype=int)
test_idxs = np.array([], dtype=int)
for label in np.unique(labels):
    matches = np.nonzero(labels == label)[0]
    np.random.shuffle(matches)
    train_idxs = np.concatenate([train_idxs, matches[:int(train_frac * len(matches))]])
    test_idxs = np.concatenate([test_idxs, matches[int(train_frac * len(matches)):]])
# In[524]:
Xc_train = Xall[train_idxs] - np.mean(Xall[train_idxs], axis=0, keepdims=True)
Xc_test = Xall[test_idxs] - np.mean(Xall[test_idxs], axis=0, keepdims=True)
U, S, Vh = np.linalg.svd(Xc_train, full_matrices=False)
# In[525]:
proj_rank = 20
train = (np.diag(S[:proj_rank]) @ U.T[:proj_rank]).T
test = Xc_test @ Vh.T[:, :proj_rank]
# In[526]:
get_ipython().run_line_magic('matplotlib', 'notebook')
sample_size = 2000  # all
fig = plt.figure(dpi=120)
ax = fig.gca(projection='3d')
plt.title("Training set responses projected onto PCA space")
# for ori in oris[[1,3]]:
for ori in oris:
    ori_ims = train[np.nonzero(labels[train_idxs]==ori)[0], :3][:sample_size // len(oris)]
    ax.scatter(ori_ims[:, 0], ori_ims[:, 1], ori_ims[:, 2], label=f"{ori}$^\circ$")
ax.view_init(elev=15, azim=50)
plt.xlabel("1st principle comp.")
plt.ylabel("2nd principle comp.")
ax.set_zlabel("3rd principle comp.")
plt.legend()
plt.show()
# In[528]:
plt.figure()
plt.scatter(np.arange(len(S)), S)
plt.xlabel("index")
plt.ylabel("$\sigma$")
plt.title("singular value spectrum")
plt.show()
# In[531]:
len(S), S[177], S[178]
# In[532]:
def mutual_info(conf_mat):
    response_freqs = np.sum(conf_mat, axis=0) / np.sum(conf_mat)
    resp_entropy = -np.sum(response_freqs * np.log2(response_freqs + 0.00001))  # bits
    print(resp_entropy)
    stim_entropy = 0
    for i in range(conf_mat.shape[0]):
        response_freqs = conf_mat[i] / np.sum(conf_mat[i])
        stim_entropy += -np.sum(response_freqs * np.log2(response_freqs + 0.00001))
    stim_entropy /= conf_mat.shape[0]
    return resp_entropy - stim_entropy

# # covariance classification
# In[533]:
avgs = []
stds = []
train_avg = np.mean(Xc_train, axis=0)
for ori in oris:
    idxs = np.nonzero(labels[train_idxs]==ori)
    avgs.append(np.mean(Xc_train[idxs] - train_avg, axis=0))
    stds.append(np.std(Xc_train[idxs] - train_avg, axis=0))


# In[534]:
def covar_classify(Xc, avgs, oris):
    pre = []
    for response in Xc:
        covars = []
        for avg in avgs:
            covars.append(np.mean(avg.T @ response))
        ori_idx = np.argmax(covars)
        pre.append(oris[ori_idx])
    return pre
# In[586]:
def covar_evaluate(Xc_train, Xc_test, labels, train_idxs, test_idxs, avgs, oris, return_minfo=False):
    fit = covar_classify(Xc_train, avgs, oris)
    train_err = np.sum(fit != labels[train_idxs]) / len(fit)
    
    pre = covar_classify(Xc_test, avgs, oris)
    err = np.sum(pre != labels[test_idxs]) / len(pre)
        
    conf = metrics.confusion_matrix(labels[test_idxs], pre, labels=oris)
    
    minfo = mutual_info(conf)
    print("mutual information:", minfo)
    
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.figure()
    plt.imshow(conf)
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(oris)))
    ax.set_yticks(np.arange(len(oris)))
    ax.set_xticklabels(oris)
    ax.set_yticklabels(oris)
    plt.colorbar()
    plt.show()
    
    if return_minfo:
        return train_err, err, minfo
    return train_err, err
# In[536]:
train_err, err = covar_evaluate(Xc_train, Xc_test, labels, train_idxs, test_idxs, avgs, oris)
print("train error:", train_err, "  test error:", err)

# ## max likelihood classification
# In[537]:
def max_likelihood_classify(Xc, avgs, stds, oris):
    pre = []
    for response in Xc:
        nlls = []
        for avg, std in zip(avgs, stds):
            nlls.append(np.mean((avg - response)**2 / (2*(std + 0.01)**2) + np.log(std + 0.01)))
        ori_idx = np.argmin(nlls)
        pre.append(oris[ori_idx])

    return pre
# In[588]:
def max_likelihood_evaluate(Xc_train, Xc_test, labels, train_idxs, test_idxs, avgs, stds, oris, return_minfo=False):
    fit = max_likelihood_classify(Xc_train, avgs, stds, oris)
    train_err = np.sum(fit != labels[train_idxs]) / len(fit)
    
    pre = max_likelihood_classify(Xc_test, avgs, stds, oris)
    err = np.sum(pre != labels[test_idxs]) / len(pre)
        
    conf = metrics.confusion_matrix(labels[test_idxs], pre, labels=oris)
    
    minfo = mutual_info(conf)
    print("mutual information:", minfo)
    
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.figure()
    plt.imshow(conf)
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(oris)))
    ax.set_yticks(np.arange(len(oris)))
    ax.set_xticklabels(oris)
    ax.set_yticklabels(oris)
    plt.colorbar()
    plt.show()
    
    if return_minfo:
        return train_err, err, minfo
    return train_err, err
# In[539]:
train_err, err = max_likelihood_evaluate(Xc_train, Xc_test, labels, train_idxs, test_idxs, avgs, stds, oris)
print("train error:", train_err, "  test error:", err)

# ## SVM
# - compare with the STA method
# - might try normalizing it so no unit is too dominant
# In[540]:
from sklearn import svm
from sklearn import metrics
# In[587]:
def svm_evaluate(train, test, labels, train_idxs, test_idxs, oris, return_minfo=False):
    clf = svm.SVC(decision_function_shape="ovo")
    clf.fit(train, labels[train_idxs])
    
    fit = clf.predict(train)
    train_err = np.sum(fit != labels[train_idxs]) / len(fit)
    
    pre = clf.predict(test)
    err = np.sum(pre != labels[test_idxs]) / len(pre)
        
    conf = metrics.confusion_matrix(labels[test_idxs], pre, labels=oris)
    
    minfo = mutual_info(conf)
    print("mutual information:", minfo)
    
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.figure()
    plt.imshow(conf)
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(oris)))
    ax.set_yticks(np.arange(len(oris)))
    ax.set_xticklabels(oris)
    ax.set_yticklabels(oris)
    plt.colorbar()
    plt.show()
    
    if return_minfo:
        return train_err, err, minfo
    return train_err, err
# In[542]:
train_err, err = svm_evaluate(train, test, labels, train_idxs, test_idxs, oris)
print("train error:", train_err, "  test error:", err)

# # Classification of all 6000 gratings
# ## Get data PCA projections
# In[553]:
num_bins = 10
stim_duration = stims.duration.min()  # minimum so that none of the recording periods are overlapping
bin_edges = np.linspace(0, stim_duration, num_bins + 1)
counts = session.presentationwise_spike_counts(bin_edges=bin_edges,
                                               stimulus_presentation_ids=stims.index,
                                               unit_ids=unit_ids)
# In[554]:
Xall = np.reshape(np.array(counts), (counts.shape[0], num_bins * len(unit_ids)))
labels = np.array(list(stims.orientation))

# train test split
train_frac = 0.7
train_idxs = np.array([], dtype=int)
test_idxs = np.array([], dtype=int)
for label in np.unique(labels):
    matches = np.nonzero(labels == label)[0]
    np.random.shuffle(matches)
    train_idxs = np.concatenate([train_idxs, matches[:int(train_frac * len(matches))]])
    test_idxs = np.concatenate([test_idxs, matches[int(train_frac * len(matches)):]])
# In[555]:
Xc_train = Xall[train_idxs] - np.mean(Xall[train_idxs], axis=0, keepdims=True)
Xc_test = Xall[test_idxs] - np.mean(Xall[test_idxs], axis=0, keepdims=True)
U, S, Vh = np.linalg.svd(Xc_train, full_matrices=False)
# In[556]:
proj_rank = 20
train = (np.diag(S[:proj_rank]) @ U.T[:proj_rank]).T
test = Xc_test @ Vh.T[:, :proj_rank]
# In[557]:
get_ipython().run_line_magic('matplotlib', 'notebook')
sample_size = 300
fig = plt.figure(dpi=120)
ax = fig.gca(projection='3d')
plt.title("Training set responses projected onto PCA space")
for ori in oris:
    ori_ims = train[np.nonzero(labels[train_idxs]==ori)[0], :3][:sample_size // len(oris)]
    ax.scatter(ori_ims[:, 0], ori_ims[:, 1], ori_ims[:, 2], label=f"{ori}$^\circ$")
ax.view_init(elev=15, azim=50)
plt.xlabel("1st principle comp.")
plt.ylabel("2nd principle comp.")
ax.set_zlabel("3rd principle comp.")
plt.legend()
plt.show()
# In[558]:
plt.figure()
plt.scatter(np.arange(len(S)), S)
plt.xlabel("index")
plt.ylabel("$\sigma$")
plt.title("singular value spectrum")
plt.show()

# # covariance classification
# In[560]:
avgs = []
stds = []
train_avg = np.mean(Xc_train, axis=0)
for ori in oris:
    idxs = np.nonzero(labels[train_idxs]==ori)
    avgs.append(np.mean(Xc_train[idxs] - train_avg, axis=0))
    stds.append(np.std(Xc_train[idxs] - train_avg, axis=0))

# In[561]:
train_err, err = covar_evaluate(Xc_train, Xc_test, labels, train_idxs, test_idxs, avgs, oris)
print("train error:", train_err, "  test error:", err)

# ## max likelihood classification
# In[562]:
train_err, err = max_likelihood_evaluate(Xc_train, Xc_test, labels, train_idxs, test_idxs, avgs, stds, oris)
print("train error:", train_err, "  test error:", err)

# ## SVM
# In[563]:
train_err, err = svm_evaluate(train, test, labels, train_idxs, test_idxs, oris)
print("train error:", train_err, "  test error:", err)

# # Natural Scene Image Classification
# ## get data and PCA projections
# In[566]:
stim_epochs = session.get_stimulus_epochs()
stim_epochs = stim_epochs[stim_epochs.stimulus_name=="natural_scenes"]
stim_epochs
# In[567]:
stims = session.get_stimulus_table(["natural_scenes"])
stims
# In[568]:
label_set = np.random.choice(np.arange(118), 6, replace=False)
filtered_stims = stims[list(frame in label_set for frame in stims.frame)]
filtered_stims.shape
# In[569]:
label_set
# Each image was shown 50 times
# In[570]:
num_bins = 10
stim_duration = filtered_stims.duration.min()  # minimum so that none of the recording periods are overlapping
bin_edges = np.linspace(0, stim_duration, num_bins + 1)
counts = session.presentationwise_spike_counts(bin_edges=bin_edges,
                                               stimulus_presentation_ids=filtered_stims.index,
                                               unit_ids=unit_ids)
# In[571]:
Xall = np.reshape(np.array(counts), (counts.shape[0], num_bins * len(unit_ids)))
labels = np.array(list(filtered_stims.frame))
# train test split
train_frac = 0.7
train_idxs = np.array([], dtype=int)
test_idxs = np.array([], dtype=int)
for label in label_set:
    matches = np.nonzero(labels == label)[0]
    np.random.shuffle(matches)
    train_idxs = np.concatenate([train_idxs, matches[:int(train_frac * len(matches))]])
    test_idxs = np.concatenate([test_idxs, matches[int(train_frac * len(matches)):]])
# In[572]:
Xc_train = Xall[train_idxs] - np.mean(Xall[train_idxs], axis=0, keepdims=True)
Xc_test = Xall[test_idxs] - np.mean(Xall[test_idxs], axis=0, keepdims=True)
U, S, Vh = np.linalg.svd(Xc_train, full_matrices=False)
# In[573]:
proj_rank = 20
train = (np.diag(S[:proj_rank]) @ U.T[:proj_rank]).T
test = Xc_test @ Vh.T[:, :proj_rank]
# In[574]:
get_ipython().run_line_magic('matplotlib', 'notebook')
sample_size = 2000
fig = plt.figure(dpi=120)
ax = fig.gca(projection='3d')
plt.title("Training set responses projected onto PCA space")
# for ori in oris[[1,3]]:
for lab in label_set:
    lab_resp = train[np.nonzero(labels[train_idxs]==lab)[0], :3][:sample_size // len(label_set)]
    ax.scatter(lab_resp[:, 0], lab_resp[:, 1], lab_resp[:, 2], label=f"{lab}")
ax.view_init(elev=15, azim=50)
plt.xlabel("1st principle comp.")
plt.ylabel("2nd principle comp.")
ax.set_zlabel("3rd principle comp.")
plt.legend()
plt.show()
# In[576]:
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
plt.scatter(np.arange(len(S)), S)
plt.xlabel("index")
plt.ylabel("$\sigma$")
plt.title("singular value spectrum")
plt.show()
# In[577]:
S[183], S[184]

# ## covariance classification
# In[578]:
avgs = []
stds = []
train_avg = np.mean(Xc_train, axis=0)
for lab in label_set:
    idxs = np.nonzero(labels[train_idxs]==lab)
    avgs.append(np.mean(Xc_train[idxs] - train_avg, axis=0))
    stds.append(np.std(Xc_train[idxs] - train_avg, axis=0))
# In[579]:
train_err, err = covar_evaluate(Xc_train, Xc_test, labels, train_idxs, test_idxs, avgs, label_set)
print("train error:", train_err, "  test error:", err)

# ## max likelihood classification
# In[580]:
train_err, err = max_likelihood_evaluate(Xc_train, Xc_test, labels, train_idxs, test_idxs, avgs, stds, label_set)
print("train error:", train_err, "  test error:", err)

# ## SVM classification
# In[581]:
train_err, err = svm_evaluate(train, test, labels, train_idxs, test_idxs, label_set)
print("train error:", train_err, "  test error:", err)

# # Classification accuracy versus number of images
# In[582]:
nums_included = np.logspace(0.4, 2.07, 10, base=10).astype(int)
nums_included
# In[589]:
errs_cov = []
errs_ml = []
errs_svm = []
minfos = []

for num_included in nums_included:
    print("NUM INCLUDED ------- ", num_included)
    label_set = np.random.choice(np.arange(118), num_included, replace=False)
    filtered_stims = stims[list(frame in label_set for frame in stims.frame)]
    
    stim_duration = filtered_stims.duration.min()  # minimum so that none of the recording periods are overlapping
    bin_edges = np.linspace(0, stim_duration, num_bins + 1)
    counts = session.presentationwise_spike_counts(bin_edges=bin_edges,
                                                   stimulus_presentation_ids=filtered_stims.index,
                                                   unit_ids=unit_ids)
    
    Xall = np.reshape(np.array(counts), (counts.shape[0], num_bins * len(unit_ids)))
    labels = np.array(list(filtered_stims.frame))

    # train test split
    train_frac = 0.7
    train_idxs = np.array([], dtype=int)
    test_idxs = np.array([], dtype=int)
    for label in label_set:
        matches = np.nonzero(labels == label)[0]
        np.random.shuffle(matches)
        train_idxs = np.concatenate([train_idxs, matches[:int(train_frac * len(matches))]])
        test_idxs = np.concatenate([test_idxs, matches[int(train_frac * len(matches)):]])
        
    Xc_train = Xall[train_idxs] - np.mean(Xall[train_idxs], axis=0, keepdims=True)
    Xc_test = Xall[test_idxs] - np.mean(Xall[test_idxs], axis=0, keepdims=True)
    U, S, Vh = np.linalg.svd(Xc_train, full_matrices=False)

    proj_rank = 20
    train = (np.diag(S[:proj_rank]) @ U.T[:proj_rank]).T
    test = Xc_test @ Vh.T[:, :proj_rank]

    avgs = []
    stds = []
    train_avg = np.mean(Xc_train, axis=0)
    for lab in label_set:
        idxs = np.nonzero(labels[train_idxs]==lab)
        avgs.append(np.mean(Xc_train[idxs] - train_avg, axis=0))
        stds.append(np.std(Xc_train[idxs] - train_avg, axis=0))

    print("COVARIANCE")
    train_err, err, minfo_cov = covar_evaluate(Xc_train, Xc_test, labels, train_idxs, test_idxs, avgs, label_set, return_minfo=True)
    errs_cov.append(err)
    print("train error:", train_err, "  test error:", err)

    print("MAX LIKELIHOOD")
    train_err, err, minfo_ml = max_likelihood_evaluate(Xc_train, Xc_test, labels, train_idxs, test_idxs, avgs, stds, label_set, return_minfo=True)
    errs_ml.append(err)
    print("train error:", train_err, "  test error:", err)

    print("SVM")
    train_err, err, minfo_svm = svm_evaluate(train, test, labels, train_idxs, test_idxs, label_set, return_minfo=True)
    errs_svm.append(err)
    print("train error:", train_err, "  test error:", err)
    
    minfos.append(max(minfo_cov, minfo_ml, minfo_svm))
# In[596]:
plt.figure()
plt.plot(nums_included, errs_cov, label="covar")
plt.plot(nums_included, errs_ml, label="max lkl")
plt.plot(nums_included, errs_svm, label="SVM")
plt.ylim([0, 1])
plt.ylabel("error rate")
plt.xlabel("number of classes")
plt.title("natural image classification performance")
plt.semilogx(base=2)
plt.legend()
plt.show()

# In[595]:
plt.figure()
plt.plot(nums_included, minfos)
plt.ylim([0, 4])
plt.ylabel("mutual information (bits)")
plt.xlabel("number of classes")
plt.title("natural image response mutual info.")
plt.semilogx(base=2)
plt.show()
