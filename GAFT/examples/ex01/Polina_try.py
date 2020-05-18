#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Find the global maximum for function: f(x) = x + 10sin(5x) + 7cos(4x)
'''

import pandas as pd
import numpy as np
import copy
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sin, cos

from GAFT.gaft import GAEngine
from GAFT.gaft.components import BinaryIndividual
from GAFT.gaft.components import Population
from GAFT.gaft.operators import TournamentSelection
from GAFT.gaft.operators import UniformCrossover
from GAFT.gaft.operators import FlipBitMutation
from GAFT.gaft.analysis.console_output import ConsoleOutput

# Analysis plugin base class.
from GAFT.gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

# Built-in best fitness analysis.
from GAFT.gaft.analysis.fitness_store import FitnessStore

from holtwintersts.holtwintersts.hw import HoltWinters


# data = pd.read_csv('./ClothingSales.csv', index_col=0)
data = pd.read_csv('./PolinaData.csv', index_col=0)
# data2 = pd.read_csv('GAFT/examples/ex01/ClothingSales.csv', index_col=0)
# print('data => ', data, '\n')
print('DATA LOADED \n')



# Define population.
indv_template = BinaryIndividual(ranges=[(0, 1), (0, 1), (0, 1)], eps=0.1)
population = Population(indv_template=indv_template, size=4).init()

# print('indv_template => ', indv_template, '\n')
# print('population => ', population, '\n')

# Create genetic operators.
selection = TournamentSelection()
# print('selection => ', selection, '\n')
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

# Create genetic algorithm engine.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[ConsoleOutput, FitnessStore])
# alpha = 1
# beta = 0
# gamma = 0.375
# best_hw = HoltWinters().fit(copy.deepcopy(data.values), [12], alpha, beta, gamma)
# mse = mean_squared_error(data2['Sales'], best_hw.fitted)
# print('best_hw =>',best_hw ,'\n')
# print('best_hw.fitted_as_dataframe() =>',best_hw.fitted.size,'\n')
# print('mse =>',mse ,'\n')
# print('mse =>',float(mse) ,'\n')
mse_mass = []

result_data = {}
# Define fitness function.
@engine.fitness_register
@engine.minimize
def fitness(indv):
    alpha, beta, gamma = indv.solution
    coeffs = []

    data_for_print_statistic = []
    best_hw = HoltWinters().fit(copy.deepcopy(data.values), [12], alpha, beta, gamma)
    mse = mean_squared_error(data['Sales'].iloc[12::], best_hw.fitted[12::])
    # for i in range(0,data['Sales'].size):
    #     print('Result FOR POLINA!: ', best_hw.fitted[i])
    #     print('Test: => \n', (data['Sales'].iloc[i]))
    # print('\n Coeffs: ', 'alpha => ', alpha, 'beta => ', beta, 'gamma => ', gamma)
    # coeffs.append(alpha)
    # coeffs.append(beta)
    # coeffs.append(gamma)
    # print('\nCoeffs: => ', coeffs)
    # # data_for_print_statistic.append(coeffs)
    # data_for_print_statistic.append(mse)
    #
    # # result_data = {data['Sales']}
    #
    # print('\nTEST DATA => ', data_for_print_statistic)
    # # print('\nResult DATA => ', result_data)
    # # print('\nResult Data size  => {}'.format(result_data.__len__()))
    # print('Best HW => ', best_hw)
    mse_mass.append(mse)
    return float(mse)



# Define on-the-fly analysis.
@engine.analysis_register
class ConsoleOutputAnalysis(OnTheFlyAnalysis):
    interval = 1
    master_only = True

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.ori_fmax)
        self.logger.info(msg)

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x = best_indv.solution
        y = engine.ori_fmax
        msg = 'Optimal solution: ({}, {})'.format(x, y)
        self.logger.info(msg)

x = np.arange(1,101)
print('Arrange => ', x)
# plt.bar()
# plt.show()
# print(mse_mass.__len__())
# for i in range(0, 30):
#     plt.bar(x, mse_mass[i])
#     plt.show

if '__main__' == __name__:
    # Run the GA engine.
    engine.run(ng=100)
    print(mse_mass.__len__())
    # for i in range (0,100):
    #     plt.bar(x,mse_mass[i])
    #     plt.show







