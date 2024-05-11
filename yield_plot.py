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

maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
yields = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']

df_yields = pdr.DataReader(yields, 'fred', start, end)

# Drop rows with missing data

df_yields = df_yields.dropna(how='any')

# %%

# 3D plot of the evolution of the term structure

dates = np.array([x.toordinal() for x in df_yields.index])

X = np.repeat(dates, len(maturities))
Y = np.tile(maturities, len(df_yields))
Z = df_yields.values.flatten()

fig = plt.figure(figsize=(6, 4.5))
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(X, Y, Z, cmap='Blues')

ax.set_xlabel('Date')
ax.set_ylabel('Maturity (years)')
ax.set_zlabel('Yield (%)')

years = np.arange(start.year, end.year + 1)
selected_years = np.linspace(start.year, end.year, 5, dtype=int)
ax.set_xticks([datetime.date(year, 1, 1).toordinal() for year in selected_years])
ax.set_xticklabels(selected_years)

plt.show()
