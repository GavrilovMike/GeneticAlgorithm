#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from gaft.examples.ex01.best_fit import best_fit

steps, variants, fits = list(zip(*best_fit))
best_step, best_v, best_f = steps[-1], variants[-1][0], fits[-1]

fig = plt.figure()


ax = fig.add_subplot()
ax.plot(steps, fits)
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')

#Plot the maximum.
# ax.scatter([best_step], [best_f], facecolor='r')
# ax.annotate(s='x: {:.2f}\ny:{:.2f}'.format(best_v, best_f),
#                                            xy=(best_step, best_f),
#                                            xytext=(best_step-0.3, best_f-0.3))


plt.show()

