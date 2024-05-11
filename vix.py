# %%

# Import the libraries

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
import matplotlib.pyplot as plt

# Visualization settings

pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.precision', 4)

# %%

# Import the data

start = datetime.date(2000, 1, 1)
end = datetime.date.today()

vol = ['VIXCLS']

df_vol = pdr.DataReader(vol, 'fred', start, end)

# Drop rows with missing data

df_vol = df_vol.dropna(how='any')

# %%

# Plot of the evolution of the fed funds rate

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.plot(df_vol, color='tab:blue', zorder=3)

ax.set_xlim(min(df_vol.index), max(df_vol.index))
ax.set_xlabel('Date')
ax.set_ylabel('%')
ax.set_title('Fed Funds Rate')
ax.grid(alpha=0.5, zorder=1)
        
fig.tight_layout()
plt.show()