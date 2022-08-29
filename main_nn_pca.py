#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import do_mpc
import pickle
import sys
sys.path.append('../')
import tensorflow as tf
import joblib


""" Choose model """
m = '4d' # either '2d' or '4d'

if m == '2d':
    from template_model_2d import template_model
elif m == '4d':
    from template_model_4d import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator



"""
Get configured do-mpc modules:
"""
model = template_model()
mpc = tf.keras.models.load_model('./models/nn_controller_pca.h5')
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)


"""
Set initial state
"""

x0_min = np.array([[20.5], [18.0], [18.0], [ 5000.0]])
x0_max = np.array([[22.5], [25.0], [25.0], [15000.0]])
x0 = np.array([[20.5], [18.0], [18.0], [5000.0]]) # for benchmarking
simulator.x0 = x0
estimator.x0 = x0


"""
Scaling
"""
# States
x_lb = np.array([[20.0, 15.0,  0.0,     0.0]])
x_ub = np.array([[23.0, 25.0, 50.0, 20000.0]])

# Control inputs
u_lb = np.array([[-1000, -1000]])
u_ub = np.array([[ 1000,  1000]])

with open(r'./data/exttemp_and_solrad_2009.pkl', 'rb') as f:
    data = pickle.load(f)
T_s = data['T']
SR_s = data['sr']
T_SR = np.hstack([T_s, SR_s])
p_lb = np.array([[-10.0,    0.0]])
p_ub = np.array([[ 30.0, 1200.0]])
T_SR = np.vstack([(p - p_lb)/(p_ub - p_lb) for p in T_SR])
T_s = []
SR_s = []
for i in range(T_SR.shape[0]-24):
    T_s.append(T_SR[i:i+25, 0])
    SR_s.append(T_SR[i:i+25, 1])

"""
Run MPC main loop:
"""

pca = joblib.load('models/pca.gz')
scaler = joblib.load('models/scaler.gz')

test_index_start = [576, 1248, 1991, 2711, 3455, 4175, 4919, 5663, 6383, 7128, 7848, 8568]
avg_violations = []
max_violations = []
total_grid_energy = []
for start_index in test_index_start:
    model = template_model()
    simulator = template_simulator(model, step_init=start_index)
    estimator = do_mpc.estimator.StateFeedback(model)
    x0 = np.array([[20.5], [18.0], [18.0], [5000.0]])
    simulator.x0 = x0
    estimator.x0 = x0
    x_res = []
    u0_res = []
    plt.ion()
    fig, ax, graphics = do_mpc.graphics.default_plot(simulator.data)


    for k in range(start_index, start_index+7*24):

        # get control input from neural network
        # scale states
        x0_scaled = (x0.T - x_lb) / (x_ub - x_lb)
        input_init = np.hstack([x0_scaled, T_s[k].reshape(1, -1), SR_s[k].reshape(1, -1)])
        input_pca = scaler.transform(input_init)
        input_pca = pca.transform(input_pca)
        u0_scaled = mpc.predict(input_pca)
        u0_real = u0_scaled * (u_ub - u_lb) + u_lb           # scale to real values
        u0_sat = np.minimum(np.maximum(u0_real, u_lb), u_ub) # ensure admissible control inputs via saturation

        # Compute the real control input variables
        P_HVAC = u0_sat[0, 0]
        if P_HVAC > 0:
            P_heat = P_HVAC
            P_cool = 0.0
        else:
            P_heat = 0.0
            P_cool = -P_HVAC

        P_bat = u0_sat[0, 1]

        u_applied = np.vstack([P_heat, P_cool, P_bat])

        y_next = simulator.make_step(u_applied)
        x0 = estimator.make_step(y_next)
        x_res.append(y_next)
        u0_res.append(u_applied)

    x_res = np.hstack(x_res)
    u0_res = np.hstack(u0_res)
    T_r = x_res[0,:]
    print(f'Start index: {start_index}')
    print(f'Occurences of T<20°C: {np.sum(x_res[0,:]<20)} times')
    print(f'Occurences of T>23°C: {np.sum(x_res[0,:]>23)} times')
    violations = []
    for T in T_r:
        if T < 20:
            violations.append(20-T)
        elif T > 23:
            violations.append(T-23)
        else:
            violations.append(0)
    avg_violation = np.mean(violations)
    max_violation = np.max(violations)
    print(f'Average T violation: {avg_violation} °C/h')
    print(f'Maximum absolute T violation: {max_violation} °C')
    total_energy = np.sum(graphics.data['_aux', 'P_grid'])/1000
    print(f"Total energy bought from/sold to the grid: {total_energy} kWh")
    print(f"Total heating energy: {np.sum(graphics.data['_u', 'P_heat']) / 1000} kWh")
    print('')
    avg_violations.append(avg_violation)
    max_violations.append(max_violation)
    total_grid_energy.append(total_energy)

    graphics.plot_results()
    graphics.reset_axes()

    np.save(f'./results/state_trajectories/nn_pca/start_index_{start_index}', x_res)

results = np.vstack([avg_violations, max_violations, total_grid_energy])
results = np.array(results)
np.save('results_nn_pca', results)

