# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 08:55:58 2018

@author: lucod
"""

import os
os.getcwd()

#%%
import matplotlib.pyplot as plt, seaborn as sns, numpy as np, os, pandas as pd
Auto = pd.read_csv("Auto.csv")
Auto.head()

#%%
'writing an ecdf function'
def cust_ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n

    return(x, y)

#%%
'generating an array of random numbers'
sim_random = np.random.random(size = 10 ** 3)

#%%
'generating an array of Bernoulli trials'
sim_bernoulli = np.random.binomial(n = 100, p = 0.05, size = 10 ** 3)

#%%
'Poisson distribution'
sim_poisson = np.random.poisson(lam = 10, size = 10 ** 3)

#%%
'normal distribution: loc = mean, scale = sd, size = no. generated'
sim_normal = np.random.normal(loc = np.mean(Auto['horsepower']), scale = np.std(Auto['horsepower']), size = 10 ** 3)

#%%
'testing the distribution of horsepower to see if normal'
x_dist, y_dist = cust_ecdf(sim_normal)
x_dist_hp, y_dist_hp = cust_ecdf(Auto['horsepower'])
_ = plt.plot(x_dist_hp, y_dist_hp, marker = '.', linestyle = 'none')
_ = plt.plot(x_dist, y_dist)
plt.show()

#%%
'running a lm on hp vs mpg and plotting the range of slopes to show where RSS minimised'
a, b = np.polyfit(Auto['horsepower'], Auto['mpg'], deg = 1)

slope_range = np.linspace(-.1, -.2, 200)

#%%
res = np.empty_like(slope_range)

for i, a in enumerate(slope_range):
    res[i] = np.sum((Auto['mpg'] - a * Auto['horsepower'] - b) ** 2)

#%%
plt.plot(slope_range, res)
plt.show()

#%%
for _ in range(50):
    bs_sample = np.random.choice(Auto['mpg'], size = len(Auto['mpg']))
    x, y  = cust_ecdf(bs_sample)
    _ = plt.plot(x, y, marker = '.', linestyle = 'none', color = 'gray', alpha = 0.1)

x, y = cust_ecdf(Auto['mpg'])
_ = plt.plot(x, y, marker = '.')

plt.margins(0.02)

plt.show()

#%%
'Taking a bootstrap of a statistic'
def boot_rep_1d(data, func):
    return func(np.random.choice(data, size = len(data)))

'Running multiple bootstrap replicates and populating a df with bs estimates'
def draw_bs_reps(data, func, no_replicates = 1):
    bs_replicates = np.empty(no_replicates)

    for i in range(no_replicates):
        bs_replicates[i] = boot_rep_1d(data, func)

    return(bs_replicates)

#%%
mpg_mean_bs = draw_bs_reps(Auto['mpg'], func = np.mean, no_replicates = 10 ** 3)

'checking distribution of the mpg mean'
sns.distplot(mpg_mean_bs)

#%%
'comparing SEM with with the bootstrap estimate'
print("Standard error of observed mpg =", round(np.std(Auto['mpg']) / np.sqrt(len(Auto['mpg'])), 4))
print("STDEV of bootstraps =", round(np.std(mpg_mean_bs), 4))

#%%
'Running replicates of linear regression parameters'

def draw_bs_pairs_linreg(x, y, size = 1):
    inds = np.arange(len(x))

    bs_slope_reps = np.empty(size)
    bs_int_reps = np.empty(size)

    for i in range(size):
        bs_inds = np.random.choice(inds, size = len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_int_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return(bs_slope_reps, bs_int_reps)

#%%
x, y = (np.polyfit(Auto['horsepower'], Auto['mpg'], 1))

#%%
slope_bs_est, int_bs_est = draw_bs_pairs_linreg(Auto['horsepower'], Auto['mpg'], size = 10 ** 3)

#%%
print(np.percentile(slope_bs_est, [2.5, 97.5]))

_ = plt.hist(slope_bs_est, bins = 50, normed = True)
_ = plt.xlabel("slope")
_ = plt.ylabel = ("PDF")

#%%
hps = np.array([40, 250])

for i in range(100):
    _  = plt.plot(hps, slope_bs_est[i] * hps + int_bs_est[i], linewidth = 0.5, alpha = 0.2, color = "red")

_ = plt.plot(Auto['horsepower'], Auto['mpg'], marker = '.', linestyle = 'none')

plt.show()

#%%
'Permutation calculations'
