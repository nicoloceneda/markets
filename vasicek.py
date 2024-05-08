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
t = np.linspace(0, T, N + 1)
num_sims = 100

# Function to simulate the Vasicek model

def vasicek(k, theta, sigma, r0, T, dt, N, num_sims):

    r = np.zeros((num_sims, N + 1))
    r[:, 0] = r0
    
    for i in range(1, N + 1):

        dr = k * (theta - r[:, i - 1]) * dt + sigma * np.random.normal(0, 1, num_sims) * np.sqrt(dt)
        r[:, i] = r[:, i - 1] + dr
    
    return r

# Simulate the paths

r = vasicek(k, theta, sigma, r0, T, dt, N, num_sims)

# Calculate empirical mean and variance at each time step

empirical_means = np.mean(r, axis=0)
empirical_std = np.std(r, axis=0)
empirical_ci_upper = empirical_means + 1.64 * empirical_std
empirical_ci_lower = empirical_means - 1.64 * empirical_std

# Calculate theoretical mean and variance at each time step

theoretical_means = r0 * np.exp(-k * t) + theta * (1 - np.exp(-k * t))
theoretical_std = np.sqrt((sigma**2 / (2 * k)) * (1 - np.exp(-2 * k * t)))
theoretical_ci_upper = theoretical_means + 1.64 * theoretical_std
theoretical_ci_lower = theoretical_means - 1.64 * theoretical_std

# Plotting the means with confidence intervals

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.plot(t, empirical_means, label='Empirical Mean', color='tab:blue', linewidth=1, zorder=5)
ax.fill_between(t, empirical_ci_lower, empirical_ci_upper, color='tab:blue', alpha=0.3, label='Empirical 90% CI', zorder=3)

ax.plot(t, theoretical_means, label='Theoretical Mean', linestyle='--', color='tab:orange', linewidth=1, zorder=2)
ax.plot(t, theoretical_ci_lower, linestyle='--', color='tab:orange', linewidth=1, zorder=2)
ax.plot(t, theoretical_ci_upper, linestyle='--', color='tab:orange', linewidth=1, zorder=2)

ax.hlines(theta, 0, T, linewidth=1, linestyles='--', colors='black', label=r'$\Theta$', zorder=2)
ax.set_xlabel('Time')
ax.set_ylabel('r(t)')
ax.set_title('Vasicek Model')
ax.set_xlim([0, T])
ax.grid(alpha=0.5, zorder=1)

fig.tight_layout()
plt.show()