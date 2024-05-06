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

df_yields = pd.read_csv('data/yields.csv', index_col=0, parse_dates=True)
df_fed_funds = pd.read_csv('data/fed_funds.csv', index_col=0, parse_dates=True)

# Time range

start = datetime.date(2000, 1, 1)
end = datetime.date.today()

# Maturities of bonds

maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]

# %%

# Animation for the evolution of the term structure

fig, ax = plt.subplots(figsize=(6, 4.5))

def animate(i):

    ax.clear()

    fine_maturities = np.linspace(min(maturities), max(maturities), 500)
    p = np.polyfit(maturities, df_yields.iloc[i], 4)
    y_fit = np.polyval(p, fine_maturities)

    ax.scatter(maturities, df_yields.iloc[i], marker='o', s=15, color='black', zorder=3)
    ax.plot(fine_maturities, y_fit, label='Yields', color='tab:blue', zorder=2)
    ax.hlines(df_fed_funds.iloc[i], -0.5, 30.5, label='Fed funds rate', linestyles='--', color='tab:orange', zorder=1)

    ax.set_title(f'US Treasury Term Structure on {df_yields.index[i].date()}')
    ax.set_xlabel('Maturity (Years)')
    ax.set_ylabel('Yield (%)')
    ax.legend(loc='upper left')
    ax.set_xlim([-0.5, 30.5])
    ax.set_ylim([0, 8])
    ax.grid(True)

ani = FuncAnimation(fig, animate, frames=len(df_yields), interval=1, repeat=False)
plt.show()
