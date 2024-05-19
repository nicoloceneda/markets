# Import the libraries

import datetime
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt

# Visualization settings

pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.precision', 4)

# Import the data

start = datetime.date(2000, 1, 1)
end = datetime.date.today()

df_yields = pdr.DataReader('DGS1MO', 'fred', start, end)

# Rename the variable

df_yields.rename(columns={'DGS1MO': '1m'}, inplace=True)

# Drop rows with missing data

yields = df_yields.dropna().values.flatten()

# Transform in decimals

yields = yields / 100

# Parameters

delta = 1
T = 252 * 50
times = np.arange(T)
r0 = yields[-1]

# Maximum Likelihood Estimation of Vasicek model parameters

def estimate_vasicek_params(yields):

    n = len(yields)
    
    num_alpha = n * np.sum(yields[1:] * yields[:-1]) - np.sum(yields[1:]) * np.sum(yields[:-1])
    den_alpha = n * np.sum(yields[:-1] ** 2) - np.sum(yields[:-1]) ** 2
    alpha_hat = num_alpha / den_alpha

    num_beta = np.sum(yields[1:] - alpha_hat * yields[:-1])
    den_beta = n * (1 - alpha_hat)
    beta_hat = num_beta / den_beta

    v2_hat = 1 / n * np.sum((yields[1:] - alpha_hat * yields[:-1] - beta_hat * (1 - alpha_hat)) ** 2) 
    
    return alpha_hat, beta_hat, v2_hat

# Estimate the parameters

alpha_hat, beta_hat, v2_hat = estimate_vasicek_params(yields)

# Convert parameters to model parameters
a = - np.log(alpha_hat) / delta
b = beta_hat * a
sigma = np.sqrt(v2_hat * 2 * a / (1 - np.exp(-2 * a * delta)))

print(f'Objective parameters:\na: {a:.4f}\nb: {b:.4f}\nsigma: {sigma:.4f}')
print(f'Objective parameters:\nSpeed: {a:.4f}\nLong run mean: {b/a:.4f}')

# Simulate the short rate process

def simulate_vasicek(r0, a, b, sigma, delta, T):

    n = int(T / delta)
    dt = delta

    rates = np.zeros(n)
    rates[0] = r0

    for t in range(1, n):

        dr = (b - a * rates[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        rates[t] = rates[t-1] + dr

    return rates

# Simulate for 1 year

simulated_rates = simulate_vasicek(r0, a, b, sigma, delta, T)

# Calculate mean and variance of r(t) over time

mean_rt = r0 * np.exp(- a * times) + b / a * (1 - np.exp(- a * times))
sigma_rt = np.sqrt(sigma ** 2 / (2 * a) * (1 - np.exp(- 2 * a * times)))

# Plot the simulated short rate process with mean and variance
plt.figure(figsize=(12, 6))
plt.plot(simulated_rates, label='Simulated Rates')
plt.plot(mean_rt, label='Mean of $r(t)$', linestyle='--')
plt.fill_between(times, mean_rt - sigma_rt, mean_rt + sigma_rt, color='orange', alpha=0.5, label='Variance of $r(t)$')
plt.title('Simulated Vasicek Short Rate Process')
plt.xlabel('Time (days)')
plt.ylabel('Short Rate')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()