import matplotlib.pyplot as plt
import pandas as pd
import copy
import numpy as np
from sklearn.metrics import mean_squared_error
from holtwintersts.holtwintersts.data_generator import univ_seasonal_gen
from holtwintersts.holtwintersts.hw import HoltWinters



# Generated data example
#data = univ_seasonal_gen([[6, 3], [12, 5]], .1, 120, 10, scale=15)

# Real data example
data = pd.read_csv('./test.csv', index_col=0)

#data['Sales'] = np.add(data['Sales'], np.random.normal(0, 5000, data.shape[0]))

smallest_params = (0, 0, 0)
smallest_mse = 100000999999999999999


# initial_hw = HoltWinters().fit(copy.deepcopy(data.values), [12], 0.8, 0.2, 0.1)
# ga_hw = HoltWinters().fit(copy.deepcopy(data.values), [12], 0.5, 0.8, 0.4)
initial_hw = HoltWinters().fit(copy.deepcopy(data.values), [12], 0.5, 0.5, 0.5)
ga_hw = HoltWinters().fit(copy.deepcopy(data.values), [12], 0.88569813, 0.9395879, 0.20468685)
initial_mse = mean_squared_error(data['Sales'].iloc[12::], initial_hw.fitted[12::])
ga_mse = mean_squared_error(data['Sales'].iloc[12::], ga_hw.fitted[12::])

print("INITIAL MSE = ", initial_mse, '\n')
print("GA MSE = ", ga_mse, '\n')
print("Delta = ", initial_mse - ga_mse, '\n')


