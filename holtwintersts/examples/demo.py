import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import copy
import datetime
import numpy as np
from sklearn.metrics import mean_squared_error
from holtwintersts.holtwintersts.data_generator import univ_seasonal_gen
from holtwintersts.holtwintersts.hw import HoltWinters



# Generated data example
#data = univ_seasonal_gen([[6, 3], [12, 5]], .1, 120, 10, scale=15)

# Real data example
# data = pd.read_csv('./ClothingSales.csv', index_col=0)
data = pd.read_csv('./PolinaData.csv', index_col=0)
data2 = pd.read_csv('./PolinaData2.csv', index_col=0)

#data['Sales'] = np.add(data['Sales'], np.random.normal(0, 5000, data.shape[0]))

smallest_params = (0, 0, 0)
smallest_mse = 100000999999999999999
count =0
for alpha in np.arange(.1, .9, .1):
    for beta in np.arange(.1, .9, .1):
        for gamma in np.arange(.1, .9, .1):
            best_hw = HoltWinters().fit(copy.deepcopy(data.values), [12], alpha, beta, gamma)
            mse = mean_squared_error(data['Sales'].iloc[12::], best_hw.fitted[12::])
            if mse < smallest_mse:
                smallest_mse = mse
                smallest_params = (copy.deepcopy(alpha), copy.deepcopy(beta), copy.deepcopy(gamma))
                print(smallest_mse)
                result = best_hw.fitted
                count += 1
                print('\n Count -> ',count,'\n best_hw.endog -> ',result,'\n')
                print(result.size)


            del best_hw
print(result.size)

print(smallest_params)

alpha = smallest_params[0]
beta = smallest_params[1]
gamma = smallest_params[2]

best_hw = HoltWinters().fit(copy.deepcopy(data2.values), [12], alpha, beta, gamma)

print ('Learning for last 10 values ', best_hw.fitted)

print('\n Alpha => ', alpha,'\n Beta => ', beta,'\n Gamma => ', gamma)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.xticks(rotation=90)
ax.plot(data)


best_hw = HoltWinters().fit(data['Sales'], [12], *smallest_params)
best_hw_results = HoltWinters

ax.plot(pd.Series(best_hw.fitted, index=data.index))

plt.show()

# best_hw = HoltWinters().fit(copy.deepcopy(data.values), [12], 0.1, 0.2, 0.3)
# smallest_params = (copy.deepcopy(0.1), copy.deepcopy(0.2), copy.deepcopy(0.3))
# mse = mean_squared_error(data['Sales'].iloc[12::], best_hw.fitted[12::])
# print(smallest_params)
# print(mse)


