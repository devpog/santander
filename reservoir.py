import os
import re

import numpy as np
import pandas as pd

from addons import *

work_dir = os.getcwd()
input_dir = os.path.join(work_dir, 'in')

sample = reservoir_sampling(input_dir, k=100000)
