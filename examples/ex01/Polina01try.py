#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Find the global maximum for function: f(x) = x + 10sin(5x) + 7cos(4x)
'''

from math import sin, cos

import sklearn

from gaft.gaft import GAEngine
from gaft.gaft.components import BinaryIndividual
from gaft.gaft.components import Population
from gaft.gaft.operators import TournamentSelection
from gaft.gaft.operators import UniformCrossover
from gaft.gaft.operators import FlipBitMutation
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


from holtwintersts.hw import HoltWinters
import matplotlib.pyplot as plt
import pandas as pd
import copy
import numpy as np
from sklearn.metrics import mean_squared_error



# Analysis plugin base class.
from gaft.gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

# Built-in best fitness analysis.
from gaft.gaft.analysis.fitness_store import FitnessStore

data = pd.read_csv('/Users/mgavrilov/Study/POLINA/gaft/examples/ex01/PolinaData.csv', index_col=0)
data = data.dropna(inplace=False)

# Define population.
indv_template = BinaryIndividual(ranges=[(0, 10),(0, 10), (0, 10), (0, 3), (0, 2), (0, 3)], eps=1)
population = Population(indv_template=indv_template, size=30).init()

# Create genetic operators.
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

# Create genetic algorithm engine.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])
smallest_mse = 99999999999
# Define fitness function.
@engine.fitness_register
@engine.dynamic_linear_scaling(target='min')
def fitness(indv):
    alpha, beta, gamma, p, d, q,  = indv.solution

    # Holt Winters Part
    best_hw = HoltWinters().fit(copy.deepcopy(data.values), [12], alpha / 10, beta / 10, gamma / 10)
    mse_hw = abs(mean_squared_error(data['Sales'].iloc[12::], best_hw.fitted[12::]))
    print("MSE HW => ", mse_hw)

    print('P => {}, D => {}, Q => {}'.format(p, d, q))
    print('INT P => {}, INT D => {}, INT Q => {}'.format(int(p),int(d), int(q)))
    # ARIMA Part
    # fit model
    train = data['Sales'][:int(0.75 * (len(data['Sales'])))]
    valid = data['Sales'][int(0.75 * (len(data['Sales']))):]
    # int_p = int(p)
    # int_d = int(d)
    # int_q = int(q)
    try:
        model = ARIMA(train, order=(int(p), int(d), int(q)))
        model_fit = model.fit(disp=1)
        # print('\n Summary => \n', model_fit.summary())
        start_index = valid.index.min()
        end_index = valid.index.max()
        # print("start_index => {}, end_index => {}".format(start_index,end_index))
        # Predictions
        predictions = model_fit.predict(start=start_index, end=end_index)
        # report performance
        mse_arima = mean_squared_error(data['Sales'][start_index:end_index], predictions)
        rmse = sqrt(mse_arima)
        print('RMSE: {}, MSE:{}'.format(rmse, mse_arima))
    except:
        mse_arima = 9999999999


    if mse_hw <= mse_arima:
        mse = mse_hw
    else:
        mse = mse_arima


    return float(mse)

# Define on-the-fly analysis.
@engine.analysis_register
class ConsoleOutputAnalysis(OnTheFlyAnalysis):
    interval = 1
    master_only = True

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.ori_fmin)
        self.logger.info(msg)

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x = best_indv.solution
        y = engine.ori_fmin
        msg = 'Optimal solution: ({}, {})'.format(x, y)
        self.logger.info(msg)

if '__main__' == __name__:
    # Run the GA engine.
    engine.run(ng=100)

