import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import GAFT.examples.ex01.Polina_try as polina



data = pd.read_csv('./Delta.csv')
# data = pd.read_csv('./PolinaData.csv', index_col=0)
mass = []
mass2 = []

# print('Data => \n', data)
# print("Only Fact Values: => \n", data['Fact'])
#
# test = data['Fact'].iloc[2] - data['Pred'].iloc[2]
# print('Test => ', test)
d = {}

# print('Size of => ', data['Fact'].size)
# for i in range(11) :
#     mse_to_raws = mean_squared_error(data['Fact'].iloc[i], data['Pred'].iloc[i])
#     print(mse_to_raws)
for i in range(0,13) :
    mse_to_raws = 0
    mse_to_raws = 1 - (abs((data['Fact'].iloc[i] - data['Pred'].iloc[i])) / data['Pred'].iloc[i])
    # mse_to_raws = mse_to_raws / (data['Fact'].iloc[i:(i+1)])
    # print("Test {i} => {mse}".format(i=i, mse=mse_to_raws))
    d[i] = mse_to_raws
    print("Dict {i} => {value}".format(i=i,value=d[i]))

# print('Data size => ', data.size/2)
# print('Test => ', mse_to_raws)
print('D => ', d)
print('D {i} => {val}'.format(i=1,val=d[1]))

plt.bar([1, 2, 3, 4, 5, 6, 7, 8 ,9 ,10, 11, 12 ,13], [d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12]])
plt.show()


for key in d:
    mass.append(key)
    if(d[key] < 0 ):
        d[key] = 0
    mass2.append(d[key])
    # print("Key {i} => {val}".format(i=key, val=mass[key]))
    # print("Val {i} => {val}".format(i=key, val=mass2[key]))


print("Mass1 {m1} => Mass 2 {m2}".format(m1=mass,m2=mass2))

plt.bar(mass, mass2)
plt.show()


print('\nResult data => \n {x}'.format(x=polina.result_data))