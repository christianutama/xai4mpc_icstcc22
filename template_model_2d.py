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
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc


def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Simple oscillating masses example with two masses and two inputs.
    # States are the position and velocitiy of the two masses.

    # States struct (optimization variables):
    T_r   = model.set_variable(var_type = '_x', var_name = 'T_r')
    # T_w   = model.set_variable(var_type = '_x', var_name = 'T_w')
    # T_e   = model.set_variable(var_type = '_x', var_name = 'T_e')
    E_bat = model.set_variable(var_type = '_x', var_name = 'E_bat')
    _x = vertcat(T_r, T_w, T_e, E_bat)

    # Input struct (optimization variables):
    P_heat = model.set_variable(var_type = '_u', var_name = 'P_heat')
    P_cool = model.set_variable(var_type = '_u', var_name = 'P_cool')
    P_grid = model.set_variable(var_type = '_u', var_name = 'P_grid')
    P_bat  = model.set_variable(var_type = '_u', var_name = 'P_bat')
    _u = vertcat(P_heat, P_cool, P_grid, P_bat)

    # Set time-varying parameters
    P_hvac = model.set_variable(var_type = '_tvp', var_name = 'd_T')
    P_bat  = model.set_variable(var_type = '_tvp', var_name = 'd_sr')
    P_grid = model.set_variable(var_type = '_tvp', var_name = 'd_int')
    _d = vertcat(d_T, d_sr, d_int)

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    P_PV = model.set_expression('P_PV', 0.5 * d_sr)

    A = np.array([[ 0.8511,  0.000],
                  [ 0.0000,  1.000]])

    B = np.array([[ 0.0035, -0.0035, 0.0000,  0.0000],
                  [ 0.0000, -0.0000, 0.0000,  1.0000]])

    E = np.array([[ 22.217,  1.7912,  42.212],
                  [ 0.0000,  0.0000,  0.0000]])
    E = 1e-3 * E


    x_next = A @ _x + B @ _u + E @ _d
    model.set_rhs('x', x_next)

    model.setup()

    return model
