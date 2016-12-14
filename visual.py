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

# Read large train set with 1,000,000 observations, without sampling
df_train = read_file(input_dir, 'train', limit=True, limit_rows=1000000)
print(df_train.shape)
# Plot histogram to see the distribution of the data
df_train.loc[df_train.age != ' NA'].age.astype(int).plot(kind='hist', bins=50, title='Age, Not sampled')


# Read in train set limiting it to just 100,000 observations, sampled with Reservoir method
sample_size = 100000
train_file = reservoir_sampling(input_dir, k=sample_size)
df_reservoir = read_file(input_dir, train_file)
print(df_reservoir.shape)

# Plot histogram to see the distribution of the data
df_reservoir.loc[df_reservoir.age != ' NA'].age.astype(int).plot(kind='hist', bins=50, title='Age, sampled with Reservoir Method')

