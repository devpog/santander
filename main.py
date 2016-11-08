import os
import re

from addons import *

import numpy as np
import pandas as pd

drop_dirs = ['.idea', '.git']

work_dir = os.getcwd()
input_dir = os.path.join(work_dir, 'in')

# Read both sets
df_train = read_file(input_dir, 'train')
df_test = read_file(input_dir, 'test')

# Change column names to X1..XN
df_train, col_map = change_column_names(df_train)




