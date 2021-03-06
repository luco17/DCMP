import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, time
from datetime import datetime as dt
from datetime import timedelta

os.getcwd()

fish = pd.read_csv("gandhi_et_al_bouts.csv", skiprows = 4)
bacteria = pd.read_csv("park_bacterial_growth.csv", skiprows = 2)
swim15 = pd.read_csv("2015_FINA.csv", skiprows = 4)
swim13 = pd.read_csv("2013_FINA.csv", skiprows =4)
parkfield = pd.read_csv("parkfield_earthquakes_1950-2017.csv", skiprows = 2)
oklahoma = pd.read_csv("oklahoma_earthquakes_1950-2017.csv", skiprows = 2)

#writing a custom ECDF function
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return(x, y)

#Writing a boostrap function
def boot_rep_1d(data, func):
    return func(np.random.choice(data, size = len(data)))

def draw_bs_reps(data, func, size = 1):
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = boot_rep_1d(data, func)
    return(bs_replicates)

#Comparing distributions of means
def diff_of_means(data1, data2):
    diff = np.mean(data1) - np.mean(data2)
    return(diff)

#Writing the perm_reps function
def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""
    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))
    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)
    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size = 1):
    """Generate multiple permutation replicates."""
    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)
    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)
        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

#Writing a function to do pairs bootstrapping with linear regression
def draw_bs_pairs_linreg(x, y, size = 1):
    inds = np.arange(len(x))

    bs_slope_reps = np.empty(size)
    bs_int_reps = np.empty(size)

    for i in range(size):
        bs_inds = np.random.choice(inds, size = len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_int_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return(bs_slope_reps, bs_int_reps)

def swap_random(a, b):
    """Randomly swap entries in two arrays."""
    # Indices to swap
    swap_inds = np.random.random(size = len(a)) < 0.5

    # Make copies of arrays a and b for output
    a_out = np.copy(a)
    b_out = np.copy(b)

    # Swap values
    b_out[swap_inds] = a[swap_inds]
    a_out[swap_inds] = b[swap_inds]

    return a_out, b_out

def pearson_r(a, b):
    corr_mat = np.corrcoef(a,b)
    return (corr_mat)[0,1]

def ecdf_formal(x, data):
    return np.searchsorted(np.sort(data), x, side='right') / len(data)

def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

#Fish data manipulation
fish['genotype'].unique()

bout_lengths_wt = fish.loc[fish['genotype'] == 'wt'].bout_length.values

bout_lengths_mut = fish.loc[fish['genotype'] == 'mut'].bout_length.values

bout_lengths_het = fish.loc[fish['genotype'] == 'het'].bout_length.values

#Generating the ECDF
x_wt, y_wt = ecdf(bout_lengths_wt)
x_mut, y_mut = ecdf(bout_lengths_mut)

#Plotting ECDFs
_ = plt.plot(x_wt, y_wt, marker = '.', linestyle = 'none')
_ = plt.plot(x_mut, y_mut, marker = '.', linestyle = 'none')
_ = plt.legend(('wt', 'mut'))
_ = plt.xlabel('active bout length (min)')
_ = plt.ylabel('ECDF')
plt.show()

#Drawing replicates from the fish means
# Compute mean active bout length
mean_wt = np.mean(bout_lengths_wt)
mean_mut = np.mean(bout_lengths_mut)

# Draw bootstrap replicates
bs_reps_wt = draw_bs_reps(bout_lengths_wt, np.mean, no_replicates = 10**4)
bs_reps_mut = draw_bs_reps(bout_lengths_mut, np.mean, no_replicates = 10**4)

# Compute 95% confidence intervals
conf_int_wt = np.percentile(bs_reps_wt, [2.5, 97.5])
conf_int_mut = np.percentile(bs_reps_mut, [2.5, 97.5])

# Print the results
print("""
wt:  mean = {0:.3f} min., conf. int. = [{1:.1f}, {2:.1f}] min.
mut: mean = {3:.3f} min., conf. int. = [{4:.1f}, {5:.1f}] min.
""".format(mean_wt, *conf_int_wt, mean_mut, *conf_int_mut))

# Compute the difference of means: diff_means_exp
diff_means_exp = np.mean(bout_lengths_het) - np.mean(bout_lengths_wt)

# Draw permutation replicates: perm_reps
perm_reps = draw_perm_reps(bout_lengths_het, bout_lengths_wt, diff_of_means, size = 10**4)

# Compute the p-value: p-val
p_val = np.sum(perm_reps >= diff_means_exp) / len(perm_reps)

# Print the result
print('p =', p_val)

"""Note that permutation testing only assesses the hypothesis that the two variables are identically distributed."""
"""A bootstrap hypo test assesses whether the means are equal."""

# Concatenate arrays: bout_lengths_concat
bout_lengths_concat = np.concatenate((bout_lengths_wt, bout_lengths_het))

# Compute mean of all bout_lengths: mean_bout_length
mean_bout_length = np.mean(bout_lengths_concat)

# Generate shifted arrays (the shifting makes the means of each equal to the concat bout_length i.e. it assumes they're the same)
wt_shifted = bout_lengths_wt - np.mean(bout_lengths_wt) + mean_bout_length
het_shifted = bout_lengths_het - np.mean(bout_lengths_het) + mean_bout_length

# Compute 10,000 bootstrap replicates from shifted arrays
bs_reps_wt = draw_bs_reps(wt_shifted, np.mean, 10 ** 4)
bs_reps_het = draw_bs_reps(het_shifted, np.mean, 10 ** 4)

# Get replicates of difference of means: bs_replicates
bs_reps = bs_reps_het - bs_reps_wt

# Compute and print p-value: p. This test is essentially saying how likely was it to get the observed diff_means if they were the same
p = np.sum(bs_reps >= diff_means_exp) / len(bs_reps)
print('p-value =', p)

#Running the analysis on bacteria area
bac_area = bacteria['bacterial area (sq. microns)'].values
bac_time = bacteria['time (hr)'].values

# Compute logarithm of the bacterial area: log_bac_area
log_bac_area = np.log(bac_area)

# Compute the slope and intercept: growth_rate, log_a0
growth_rate, log_a0 = np.polyfit(bac_time, log_bac_area, 1)

# Draw 10,000 pairs bootstrap replicates: growth_rate_bs_reps, log_a0_bs_reps
growth_rate_bs_reps, log_a0_bs_reps = draw_bs_pairs_linreg(bac_time, log_bac_area, size = 10 ** 4)

# Compute confidence intervals: growth_rate_conf_int
growth_rate_conf_int = np.percentile(growth_rate_bs_reps, [2.5, 97.5])

# Print the result to the screen
print("""
Growth rate: {0:.4f} sq. µm/hour
95% conf int: [{1:.4f}, {2:.4f}] sq. µm/hour
""".format(growth_rate, *growth_rate_conf_int))

# Plot data points in a semilog-y plot with axis labeles
_ = plt.semilogy(bac_time, bac_area, marker='.', linestyle='none')

# Generate x-values for the bootstrap lines: t_bs
t_bs = np.array([0, 14])

# Plot the first 100 bootstrap lines
for i in range(100):
    y = np.exp(growth_rate_bs_reps[i] * t_bs + log_a0_bs_reps[i])
    _ = plt.semilogy(t_bs, y, linewidth = .5, alpha = .05, color = 'red')

# Label axes and show plot
_ = plt.xlabel('time (hr)')
_ = plt.ylabel('area (sq. µm)')
plt.show()

"""The predicted line does not stray far from the experimental line"""

############## Working with the FINA swimming data ###################
swim15.head()

swim15.stroke.unique()
swim15.distance.unique()
swim15['round'].unique()

mens_200_free_heats = swim15.loc[(swim15['gender'] == "M") & (swim15['stroke'] == "FREE") & (swim15['distance'] == 200) & (swim15['round'] == "PRE") & (swim15['split'] == 4)].cumswimtime.values

x, y = ecdf(mens_200_free_heats)

plt.plot(x, y, marker = ".", linestyle = "none")
plt.xlabel("Swim time (s)")
plt.ylabel("ECDF")
plt.show()
# Compute mean and median swim times
mean_time = np.mean(mens_200_free_heats)
median_time = np.median(mens_200_free_heats)

# Draw 10,000 bootstrap replicates of the mean and median
bs_reps_mean = draw_bs_reps(mens_200_free_heats, np.mean, 10**4)
bs_reps_median = draw_bs_reps(mens_200_free_heats, np.median, 10**4)

# Compute the 95% confidence intervals
conf_int_mean = np.percentile(bs_reps_mean, [2.5, 97.5])
conf_int_median = np.percentile(bs_reps_median, [2.5, 97.5])

# Print the result to the screen
print("""
mean time: {0:.2f} sec.
95% conf int of mean: [{1:.2f}, {2:.2f}] sec.

median time: {3:.2f} sec.
95% conf int of median: [{4:.2f}, {5:.2f}] sec.
""".format(mean_time, *conf_int_mean, median_time, *conf_int_median))

women_swim_df = swim15.loc[(swim15['gender'] == "F") & (swim15['stroke'] != "MEDLEY") & (swim15['distance'].isin([100, 50, 200])) & (swim15['round'].isin(['SEM', 'FIN'])) & (swim15['splitdistance'] == swim15['distance'])]
women_swim_df.head(n = 5)
women_swim_df.columns
women_swim_df = women_swim_df[['athleteid', 'stroke', 'distance', 'lastname', 'cumswimtime', 'round']]
len(women_swim_df.athleteid.unique())

women_swim_df_fin = women_swim_df.loc[(women_swim_df['round'] == 'FIN')]
women_swim_df_sem = women_swim_df.loc[(women_swim_df['round'] == 'SEM')]

women_swim_df_w = women_swim_df_fin.merge(women_swim_df_sem, how = 'left', on = ['athleteid', 'stroke', 'distance', 'lastname'])

women_swim_df_w = women_swim_df_w.rename(index = str, columns = {"cumswimtime_x" : "final_swimtime", "cumswimtime_y" : "semi_swimtime"})
women_swim_df_w = women_swim_df_w[['athleteid', 'stroke', 'distance', 'lastname', 'final_swimtime', 'semi_swimtime']]

women_swim_df_w.head()
women_swim_df_w.shape

final_times = women_swim_df_w['final_swimtime'].values
semi_times = women_swim_df_w['semi_swimtime'].values

# Compute fractional difference in time between finals and semis
f = (semi_times - final_times) / semi_times

# Generate x and y values for the ECDF: x, y
x, y = ecdf(f)

# Make a plot of the ECDF
plt.plot(x, y, marker = ".", linestyle = "none")
_ = plt.xlabel('f')
_ = plt.ylabel('ECDF')
plt.show()

# Mean fractional time difference: f_mean
f_mean = np.mean(f)

# Get bootstrap reps of mean: bs_reps
bs_reps = dcst.draw_bs_reps(f, np.mean, size = 10**4)

# Compute confidence intervals: conf_int
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Report
print("""
mean frac. diff.: {0:.5f}
95% conf int of mean frac. diff.: [{1:.5f}, {2:.5f}]""".format(f_mean, *conf_int))

 # Set up array of permutation replicates
perm_reps = np.empty(shape = 10**3)

for i in range(1000):
    # Generate a permutation sample
    semi_perm, final_perm = swap_random(semi_times, final_times)

    # Compute f from the permutation sample
    f = (semi_perm - final_perm) / semi_perm

    # Compute and store permutation replicate
    perm_reps[i] = np.mean(f)

# Compute and print p-value
print('p =', np.sum(perm_reps >= f_mean) / 1000)

#Working the women's 800 free heats#
free_800_w = swim15.loc[(swim15['gender'] == "F") & (swim15['stroke'] == "FREE") & (swim15['distance'] == 800) & (swim15['round'].isin(['PRE'])) & (~swim15['split'].isin([1,2,15,16]))]
swim15.columns.values
free_800_w = free_800_w[['split', 'splitswimtime']]

splits = np.reshape(free_800_w['splitswimtime'].values, (-1, 12))
split_number = free_800_w['split'].unique()

len(split_number)
len(splits)

# Plot the splits for each swimmer
for i in splits:
    _ = plt.plot(split_number, i, lw = 1, color = 'lightgray')

# Compute the mean split times
mean_splits = np.mean(splits, axis = 0)

# Plot the mean split times
_ = plt.plot(split_number, mean_splits, marker = '.', linewidth = 3, markersize = 12)

# Label axes and show plot
_ = plt.xlabel('split number')
_ = plt.ylabel('split time (s)')
plt.show()
############### Running linear regression analysis ##############
# Perform regression
slowdown, split_3 = np.polyfit(x = split_number, y = mean_splits, deg = 1)

# Compute pairs bootstrap
bs_reps, _ = draw_bs_pairs_linreg(split_number, mean_splits, size = 10**4)

# Compute confidence interval
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Plot the data with regressions line
_ = plt.plot(split_number, mean_splits, marker = '.', linestyle = 'none')
_ = plt.plot(split_number, slowdown * split_number + split_3, linestyle = '-')

# Label axes and show plot
_ = plt.xlabel('split number')
_ = plt.ylabel('split time (s)')
plt.show()

# Print the slowdown per split
print("""
mean slowdown: {0:.3f} sec./split
95% conf int of mean slowdown: [{1:.3f}, {2:.3f}] sec./split""".format(
    slowdown, *conf_int))

###Hypo testing the slowdown to see if the correlation is by chance###
# Observed correlation
rho = pearson_r(split_number, mean_splits)
rho

# Initialize permutation reps
perm_reps_rho = np.empty(shape = 10 ** 4)

# Make permutation reps
for i in range(10000):
    # Scramble the split number array
    scrambled_split_number = np.random.permutation(split_number)

    # Compute the Pearson correlation coefficient
    perm_reps_rho[i] = pearson_r(scrambled_split_number, mean_splits)

# Compute and print p-value
p_val = np.sum(perm_reps_rho >= rho) / len(perm_reps_rho)
print('p =', p_val)

### Parkfield Seismology Analysis ###

# EDA to look at the magnitudes of earthquakes in Parkfield
parkfield.columns.values
mags = parkfield.mag.values

# Label axes and show plot
_ = plt.plot(*ecdf(mags), marker = '.', linestyle = 'none')
_ = plt.xlabel('magnitude')
_ = plt.ylabel('ECDF')
plt.show()

#Writing a function to compute the b-value of an earthquake region#
def b_value(mags, mt, perc = [2.5, 97.5], n_reps = None):
    """Compute the b-value and optionally its confidence interval."""
    # Extract magnitudes above completeness threshold: m
    m = mags[mags >= mt]

    # Compute b-value: b
    b = (np.mean(m) - mt) * np.log(10)

    # Draw bootstrap replicates
    if n_reps is None:
        return b
    else:
        m_bs_reps = draw_bs_reps(m, np.mean, size = n_reps)

        # Compute b-value from replicates: b_bs_reps
        b_bs_reps = (m_bs_reps - mt) * np.log(10)

        # Compute confidence interval: conf_int
        conf_int = np.percentile(b_bs_reps, perc)

        return b, conf_int

### Comparing Parkfield earthquakes vs normal distribution at mag 3 ###
# Compute b-value and confidence interval
mt = 3

b, conf_int = b_value(mags, mt, perc=[2.5, 97.5], n_reps = 10**4)

# Generate samples to for theoretical ECDF
m_theor = np.random.exponential(b/np.log(10), size = 10 **5 ) + mt

# Plot the theoretical CDF
_ = plt.plot(*ecdf(m_theor))

# Plot the ECDF (slicing mags >= mt)
_ = plt.plot(*ecdf(mags[mags >= mt]), marker = '.', linestyle = 'none')

# Pretty up and show the plot
_ = plt.xlabel('magnitude')
_ = plt.ylabel('ECDF')
_ = plt.xlim(2.8, 6.2)
plt.show()

# Report the results
print("""
b-value: {0:.2f}
95% conf int: [{1:.2f}, {2:.2f}]""".format(b, *conf_int))

#manual input of earthquake time gaps#
time_gap = np.array([24.65, 20.076, 21.018, 12.246, 32.054, 38.253])

# Compute the mean time gap: mean_time_gap
mean_time_gap = np.mean(time_gap)

# Standard deviation of the time gap: std_time_gap
std_time_gap = np.std(time_gap)

# Generate theoretical Exponential distribution of timings: time_gap_exp
time_gap_exp = np.random.exponential(mean_time_gap, size = 10**4)

# Generate theoretical Normal distribution of timings: time_gap_norm
time_gap_norm = np.random.normal(loc = mean_time_gap, scale = std_time_gap, size = 10**4)

# Plot theoretical CDFs
_ = plt.plot(*ecdf(time_gap_exp))
_ = plt.plot(*ecdf(time_gap_norm))

# Plot Parkfield ECDF
_ = plt.plot(*ecdf(time_gap), marker = '.', linestyle = 'none')

# Add legend
_ = plt.legend(('Exp.', 'Norm.', 'Actual'), loc='upper left')

# Label axes, set limits and show plot
_ = plt.xlabel('time gap (years)')
_ = plt.ylabel('ECDF')
_ = plt.xlim(-10, 50)
plt.show()

# Calculating when the next earthquake will be
today = 2018.9064
last_quake = 2004.74

# Draw samples from the Exponential distribution: exp_samples
exp_samples = np.random.exponential(scale = mean_time_gap, size = 10**5)

# Draw samples from the Normal distribution: norm_samples
norm_samples = np.random.normal(loc = mean_time_gap, scale = std_time_gap, size = 10**5)

# No earthquake as of today, so only keep samples that are long enough
exp_samples = exp_samples[exp_samples > today - last_quake]
norm_samples = norm_samples[norm_samples > today - last_quake]

# Compute the confidence intervals with medians
conf_int_exp = np.percentile(exp_samples, [2.5, 50, 97.5]) + last_quake
conf_int_norm = np.percentile(norm_samples, [2.5, 50, 97.5]) + last_quake

# Print the results
print('Exponential:', conf_int_exp)
print('     Normal:', conf_int_norm)

def ks_stat(data1, data2):
    # Compute ECDF from data: x, y
    x, y = ecdf(data1)

    # Compute corresponding values of the target CDF
    cdf = ecdf_formal(x, data2)

    # Compute distances between concave corners and CDF
    D_top = y - cdf

    # Compute distance between convex corners and CDF
    D_bottom = cdf - y + 1/len(data1)

    return np.max((D_top, D_bottom))

def draw_ks_reps(n, f, args = (), size = 10**4, n_reps = 10**4):
    # Generate samples from target distribution
    x_f = f(*args, size = size)

    # Initialize K-S replicates
    reps = np.empty(n_reps)

    # Draw replicates
    for i in range(n_reps):
        # Draw samples for comparison
        x_samp = f(*args, size = n)

        # Compute K-S statistic
        reps[i] = ks_stat(x_samp, x_f)

    return reps

# Draw target distribution
x_f = np.random.exponential(mean_time_gap, size=10000)

# Compute K-S stat: d
d = ks_stat(time_gap, x_f)

# Draw K-S replicates: reps
reps = draw_ks_reps(len(time_gap), np.random.exponential,
                         args=(mean_time_gap,), size=10000, n_reps=10000)

# Compute and print p-value
p_val = np.sum(reps >= d) / 10000
print('p =', p_val)


### Oklahomo Seismology Analysis ###
oklahoma.columns.values
#Converting time column to datetimes
oklahoma['time'] = pd.to_datetime(oklahoma['time'])
oklahoma.time.head()

oklahoma['date_decimal'] = oklahoma['time'].apply(lambda x: toYearFraction(x))

mags = oklahoma.loc[(oklahoma['date_decimal'] > 1980) & (oklahoma['date_decimal'] < 2017.5), ['mag']].values
time = oklahoma.loc[(oklahoma['date_decimal'] > 1980) & (oklahoma['date_decimal'] < 2017.5), ['date_decimal']].values

#EDA#
# Plot time vs. magnitude
_ = plt.plot(time, mags, marker = '.', linestyle = 'none', alpha = .3)

# Label axes and show the plot
_ = plt.xlabel('time (year)')
_ = plt.ylabel('magnitude')
plt.show()

#Taking the daily date between earthquakes over mag 3 from 1980 - 2009 and then from 2010 to mid 2017#

dt_pre = oklahoma.loc[(oklahoma['date_decimal'] > 1980) & (oklahoma['date_decimal'] < 2010) & (oklahoma['mag'] >= 3), ['time']]
dt_pre = ((dt_pre['time'].diff().dropna()) / timedelta (days = 1)).values

dt_post = oklahoma.loc[(oklahoma['date_decimal'] >= 2010) & (oklahoma['date_decimal'] < 2017.5) & (oklahoma['mag'] >= 3), ['time']]
dt_post = ((dt_post['time'].diff().dropna()) / timedelta (days = 1)).values

# Compute mean interearthquake time
mean_dt_pre = np.mean(dt_pre)
mean_dt_post = np.mean(dt_post)

# Draw 10,000 bootstrap replicates of the mean
bs_reps_pre = draw_bs_reps(dt_pre, np.mean, size = 10**4)
bs_reps_post = draw_bs_reps(dt_post, np.mean, size = 10**4)

# Compute the confidence interval
conf_int_pre = np.percentile(bs_reps_pre, [2.5, 97.5])
conf_int_post = np.percentile(bs_reps_post, [2.5, 97.5])

# Print the results
print("""1980 through 2009
mean time gap: {0:.2f} days
95% conf int: [{1:.2f}, {2:.2f}] days""".format(mean_dt_pre, *conf_int_pre))

print("""
2010 through mid-2017
mean time gap: {0:.2f} days
95% conf int: [{1:.2f}, {2:.2f}] days""".format(mean_dt_post, *conf_int_post))
# Compute the observed test statistic
mean_dt_diff = mean_dt_pre - mean_dt_post

# Shift the post-2010 data to have the same mean as the pre-2010 data
dt_post_shift = dt_post - mean_dt_post + mean_dt_pre

# Compute 10,000 bootstrap replicates from arrays
bs_reps_pre = draw_bs_reps(dt_pre, np.mean, size = 10**4)
bs_reps_post = draw_bs_reps(dt_post_shift, np.mean, size = 10**4)

# Get replicates of difference of means
bs_reps = bs_reps_pre - bs_reps_post

# Compute and print the p-value
p_val = np.sum(bs_reps >= mean_dt_diff) / 10000
print('p =', p_val)

##Comparing magnitudes pre and post fracking##
mags_pre = mags[time < 2010]
mags_post = mags[time >= 2010]

# Generate ECDFs
_ = plt.plot(*ecdf(mags_pre), marker = '.', linestyle = 'none')

_ = plt.plot(*ecdf(mags_post), marker = '.', linestyle = 'none')

# Label axes and show plot
_ = plt.xlabel('magnitude')
_ = plt.ylabel('ECDF')
plt.legend(('1980 though 2009', '2010 through mid-2017'), loc='upper left')
plt.show()

# Compute b-value and confidence interval for pre-2010
b_pre, conf_int_pre = b_value(mags_pre, mt, perc = [2.5, 97.5], n_reps = 10**4)

# Compute b-value and confidence interval for post-2010
b_post, conf_int_post = b_value(mags_post, mt, perc = [2.5, 97.5], n_reps = 10**4)

# Report the results
print("""
1980 through 2009
b-value: {0:.2f}
95% conf int: [{1:.2f}, {2:.2f}]

2010 through mid-2017
b-value: {3:.2f}
95% conf int: [{4:.2f}, {5:.2f}]
""".format(b_pre, *conf_int_pre, b_post, *conf_int_post))

# Only magnitudes above completeness threshold
mags_pre = mags_pre[mags_pre >= mt]
mags_post = mags_post[mags_post >= mt]

# Observed difference in mean magnitudes: diff_obs
diff_obs = np.mean(mags_post) - np.mean(mags_pre)

# Generate permutation replicates: perm_reps
perm_reps = draw_perm_reps(mags_post, mags_pre, diff_of_means, size = 10**4)

# Compute and print p-value
p_val = np.sum(perm_reps < diff_obs) / 10000
print('p =', p_val)
