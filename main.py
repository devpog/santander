import os
import re

import numpy as np
import pandas as pd
import seaborn as sb

from datetime import datetime
from dateutil.parser import parse

from matplotlib import pyplot as plt

from addons import *

work_dir = os.getcwd()
input_dir = os.path.join(work_dir, 'in')

# Read in train set
train_file = reservoir_sampling(input_dir, k=100000)
df_train = read_file(input_dir, train_file)

# Change column names to X1..XN
df_train, col_map = change_column_names(df_train)

# Get a random 10%-sample of the original train set
df = df_train.copy()

# Cast X1, X7 into datetime
to_date = ['X1', 'X7', 'X11']
print("Cast {} into datetime".format(' '.join(to_date)))
df.loc[:, to_date] = df.loc[:, to_date].apply(lambda c: pd.to_datetime(c))

# Create 2 additional columns signifying the month of action and the month of customers joining
df['X49'] = pd.DatetimeIndex(df.loc[:, 'X1']).month.astype(int)
df['X50'] = pd.DatetimeIndex(df.loc[:, 'X7']).month.astype(int)

# Age distribution
df.loc[:, 'X6'] = df.loc[:, 'X6'].apply(lambda c: pd.to_numeric(c, errors='coerce'))
# Display histogram
df['X6'].hist(bins=80)
df.loc[df.X6 < 18, 'X6'] = df.loc[(df.X6 >= 18) & (df.X6 <= 30), 'X6'].mean(skipna=True)
df.loc[df.X6 > 80, 'X6'] = df.loc[(df.X6 >= 30) & (df.X6 <= 80), 'X6'].mean(skipna=True)

# Munging customer's seniority
df.loc[:, 'X9'] = pd.to_numeric(df.X9, errors='coerce')
df.loc[:, 'X9'] = df['X9'].min()

# Munging dates when customers joined the company
dates = df.loc[:, 'X7'].sort_values().reset_index()
med_date = np.median(dates.index.values).round()
df.loc[df.X7.isnull(), 'X7'] = dates.loc[med_date, 'X7']
# Assigned the most frequent to unassigned
df.loc[df.X10.isnull(), 'X10'] = max(df.X10.value_counts().items(), key=lambda x: x[1])[0]

# Create map of province names/codes, dropping X21 - name
# and assigning the max()+1 map's ID to NaN/UNKNOWN
df.loc[df.X21.isnull(), 'X21'] = 'UNKNOWN'
df.loc[df.X20.isnull(), 'X20'] = int(df.loc[:, 'X20'].max()) + 1
province_map = dict((int(l[0]), ''.join(map(str, l[1:]))) for l in df.loc[:, ['X20', 'X21']].values if pd.notnull(l[0]))
"""
to_drop = ['X21']
df = df.drop(to_drop, axis=1)
"""

# Massage X23 (renta) assigning median by providence to missing values
incomes = df.loc[df.X23.notnull(),:].groupby('X21').agg({'X23':{'MedianIncome': np.median,                                                                'Mean': np.mean}})
incomes.sort_values(by=('X23', 'MedianIncome'), inplace=True)
incomes.reset_index(inplace=True)
incomes.X21 = incomes.X21.astype('category', categories=[i for i in df.X21.unique()], ordered=False)
with sb.axes_style():
    h = sb.factorplot(data=incomes,
                      x='X21',
                      y=('X23', 'MedianIncome'),
                      order=(i for i in incomes.X21),
                      scale=1.0,
                      linestyle='None')
plt.xticks(rotation=90)
plt.ylabel('Median/Mean Income')
plt.xlabel('City')
plt.ylim(0, 180000)
plt.yticks(range(0, 180000, 40000))

