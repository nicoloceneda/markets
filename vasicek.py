# %%

# Import the libraries

import numpy as np
import matplotlib.pyplot as plt

# Parameters

k = 1.5
theta = 0.05
sigma = 0.02
r0 = 0.02
T = 5
dt = 0.01
N = int(T / dt)
times = np.linspace(0, T, N + 1)
num_sims = 100
maturities = np.linspace(0.01, 30, 20)

# %%

# Function to simulate the Vasicek model

def vasicek(k, theta, sigma, r0, dt, N, num_sims):

    r = np.zeros((num_sims, N + 1))
    r[:, 0] = r0
    
    for i in range(1, N + 1):

        dr = k * (theta - r[:, i - 1]) * dt + sigma * np.random.normal(0, 1, num_sims) * np.sqrt(dt)
        r[:, i] = r[:, i - 1] + dr
    
    return r

# Function to calculate bond prices

def bond_prices(t, T, r, k, theta, sigma):

    B = 1 / k * (1 - np.exp(-k * (T - t)))
    A = np.exp((theta - sigma ** 2 / (2 * k ** 2)) * (B - T + t) - sigma ** 2 * B ** 2 / (4 * k))

    return A * np.exp(-B * r)

# %%

# Simulate the paths

r = vasicek(k, theta, sigma, r0, dt, N, num_sims)

# Calculate empirical mean and variance at each time step

empirical_means = np.mean(r, axis=0)
empirical_std = np.std(r, axis=0)
empirical_ci_upper = empirical_means + 1.64 * empirical_std
empirical_ci_lower = empirical_means - 1.64 * empirical_std

# Calculate theoretical mean and variance at each time step

theoretical_means = r0 * np.exp(-k * times) + theta * (1 - np.exp(-k * times))
theoretical_std = np.sqrt((sigma**2 / (2 * k)) * (1 - np.exp(-2 * k * times)))
theoretical_ci_upper = theoretical_means + 1.64 * theoretical_std
theoretical_ci_lower = theoretical_means - 1.64 * theoretical_std

# Plot the means and confidence intervals

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.plot(times, empirical_means, color='tab:blue', linewidth=1, zorder=4)
ax.fill_between(times, empirical_ci_lower, empirical_ci_upper, color='tab:blue', alpha=0.3, zorder=3)
ax.plot(times, theoretical_means, linestyle='--', color='tab:orange', linewidth=1, zorder=2)
ax.plot(times, theoretical_ci_lower, linestyle='--', color='tab:orange', linewidth=1, zorder=2)
ax.plot(times, theoretical_ci_upper, linestyle='--', color='tab:orange', linewidth=1, zorder=2)
ax.hlines(theta, 0, T, linestyles='--', colors='black', linewidth=1, zorder=2)
ax.set_xlabel('Time')
ax.set_ylabel('r(t)')
ax.set_title('Vasicek Model')
ax.set_xlim([0, T])
ax.grid(alpha=0.5, zorder=1)

fig.tight_layout()
plt.show()

# %%

# Compute the bond prices for each maturity

bond_prices_array = np.zeros((N + 1, len(maturities)))

for idx, maturity in enumerate(maturities):

    bond_prices_array[:, idx] = bond_prices(0, maturity, empirical_means, k, theta, sigma)

# Plot the bond price surface

X = np.repeat(times, len(maturities))
Y = np.tile(maturities, len(bond_prices_array))
Z = bond_prices_array.flatten()

fig = plt.figure(figsize=(6, 4.5))
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(X, Y, Z, cmap='autumn_r', linewidth=1)

ax.set_xlabel('Time')
ax.set_ylabel('Maturity (years)')
ax.set_zlabel('Bond Price')

ax.view_init(elev=20, azim=125)
plt.show()
