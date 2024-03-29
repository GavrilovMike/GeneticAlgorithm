#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Find the global maximum for function: f(x) = x + 10sin(5x) + 7cos(4x)
'''

from math import sin, cos

from GAFT.gaft import GAEngine
from GAFT.gaft.components import BinaryIndividual
from GAFT.gaft.components import Population
from GAFT.gaft.operators import TournamentSelection
from GAFT.gaft.operators import UniformCrossover
from GAFT.gaft.operators import FlipBitMutation

# Analysis plugin base class.
from GAFT.gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

# Built-in best fitness analysis.
from GAFT.gaft.analysis.fitness_store import FitnessStore

# Define population.
indv_template = BinaryIndividual(ranges=[(1, 10)], eps=0.001)
population = Population(indv_template=indv_template, size=30).init()

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
                  analysis=[FitnessStore])

# Define fitness function.
# @engine.fitness_register
@engine.fitness_register
@engine.minimize
def fitness(indv):
    x, = indv.solution
    return x + 10*sin(5*x) + 7*cos(4*x)

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

if '__main__' == __name__:
    # Run the GA engine.
    engine.run(ng=100)

