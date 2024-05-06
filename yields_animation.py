# %%

# Import the libraries

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Visualization settings

pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.precision', 4)

# %%

# Import the data

start = datetime.date(2000, 1, 1)
end = datetime.date.today()

maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
yields = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
fed_funds = ['DFF']

df_yields = pdr.DataReader(yields, 'fred', start, end)
df_fed_funds = pdr.DataReader(fed_funds, 'fred', start, end)

# Drop rows with missing data

df_yields = df_yields.dropna(how='any')
df_fed_funds = df_fed_funds[df_fed_funds.index.isin(df_yields.index)]

# %%

# Animation for the evolution of the term structure

fig, ax = plt.subplots(figsize=(6, 4.5))

def animate(i):

    ax.clear()

    fine_maturities = np.linspace(min(maturities), max(maturities), 500)
    p = np.polyfit(maturities, df_yields.iloc[i], 4)
    y_fit = np.polyval(p, fine_maturities)

    ax.scatter(maturities, df_yields.iloc[i], marker='o', s=15, color='black', zorder=3)
    ax.plot(fine_maturities, y_fit, label='Fitted Polynomial', color='tab:blue', zorder=2)
    ax.hlines(df_fed_funds.iloc[i], -0.5, 30.5, linestyles='--', color='tab:orange', zorder=1)

    ax.set_title(f'US Treasury Term Structure on {df_yields.index[i].date()}')
    ax.set_xlabel('Maturity (Years)')
    ax.set_ylabel('Yield (%)')
    ax.set_xlim([-0.5, 30.5])
    ax.set_ylim([0, 8])
    ax.grid(True)

ani = FuncAnimation(fig, animate, frames=len(df_yields), interval=1, repeat=False)
plt.show()

# %%

# Compute the rolling volatility of the term structure

df_vol_252 = df_yields.rolling(window=252).std()
df_vol_21 = df_yields.rolling(window=21).std()

# Drop rows with missing data

df_vol_252 = df_vol_252.dropna(how='any')
df_vol_21 = df_vol_21[df_vol_21.index.isin(df_vol_252.index)]

# Animation for the volatility of the term structure

fig, ax = plt.subplots(figsize=(6, 4.5))

def animate(i):

    ax.clear()

    fine_maturities = np.linspace(min(maturities), max(maturities), 500)
    p = np.polyfit(maturities, df_vol_252.iloc[i], 4)
    y_fit_252 = np.polyval(p, fine_maturities)
    p = np.polyfit(maturities, df_vol_21.iloc[i], 4)
    y_fit_21 = np.polyval(p, fine_maturities)

    ax.scatter(maturities, df_vol_252.iloc[i], marker='o', s=15, color='black', zorder=3)
    ax.plot(fine_maturities, y_fit_252, label='252 days', color='tab:blue', zorder=2)

    ax.scatter(maturities, df_vol_21.iloc[i], marker='o', s=15, color='black', zorder=3)
    ax.plot(fine_maturities, y_fit_21, label='21 days', color='tab:orange', zorder=2)

    ax.set_title(f'US Treasury Volatility on {df_vol_252.index[i].date()}')
    ax.set_xlabel('Maturity (Years)')
    ax.set_ylabel('Volatility (%)')
    ax.set_xlim([-0.5, 30.5])
    ax.set_ylim([0, 2])
    ax.grid(True)

ani = FuncAnimation(fig, animate, frames=len(df_vol_252), interval=1, repeat=False)
plt.show()