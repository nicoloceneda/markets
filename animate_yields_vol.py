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

df_vol_252 = pd.read_csv('data/vol_252.csv', index_col=0, parse_dates=True)
df_vol_21 = pd.read_csv('data/vol_21.csv', index_col=0, parse_dates=True)

# Time range

start = datetime.date(2000, 1, 1)
end = datetime.date.today()

# Maturities of bonds

maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]

# %%

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
    ax.legend(loc='upper left')
    ax.set_xlim([-0.5, 30.5])
    ax.set_ylim([0, 1.75])
    ax.grid(True)

ani = FuncAnimation(fig, animate, frames=len(df_vol_252), interval=1, repeat=False)
plt.show()