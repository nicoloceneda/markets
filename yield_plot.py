# %%

# Import the libraries

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization settings

pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.precision', 4)

# %%

# Import the data

start = datetime.date(2000, 1, 1)
end = datetime.date.today()

maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
yields = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']

df_yields = pdr.DataReader(yields, 'fred', start, end)

# Rename columns

new_names = {
    'DGS1MO': '1m',
    'DGS3MO': '3m',
    'DGS6MO': '6m',
    'DGS1': '1y',
    'DGS2': '2y',
    'DGS3': '3y',
    'DGS5': '5y',
    'DGS7': '7y',
    'DGS10': '10y',
    'DGS20': '20y',
    'DGS30': '30y'
}
df_yields.rename(columns=new_names, inplace=True)

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

fig.tight_layout()
plt.show()

# %%

# Compute the average yield for each maturity and year

average_yields = df_yields.groupby(df_yields.index.year).mean().transpose()

# Plot the heatmap of average yields

fig, ax = plt.subplots(figsize=(6, 4.5))
sns.heatmap(average_yields, cmap='coolwarm', annot=False, linewidths=.5, linecolor='black', ax=ax)

ax.set_title('Average Yield')
ax.set_xlabel('Year')
ax.set_ylabel('Maturity (years)')
plt.gca().invert_yaxis()

fig.tight_layout()
plt.show()
