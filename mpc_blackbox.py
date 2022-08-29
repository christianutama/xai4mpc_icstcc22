from template_model_4d import template_model
from template_mpc import template_mpc_shap
from template_simulator import template_simulator_shap
from casadi.tools import *
import do_mpc


def extract_T_sr(X, p_lb=np.array([[-10.0,    0.0]]), p_ub=np.array([[30.0, 1200.0]])):
    """ Extract T and SR from ML training data (X)"""
    T_data = X[4:29].reshape(-1, 1)
    sr_data = X[29:].reshape(-1, 1)
    T_SR = np.hstack([T_data, sr_data])
    # convert to original values
    T_SR = np.vstack([p_lb+p*(p_ub - p_lb) for p in T_SR])
    return T_SR


def extract_state_vars(X,
                       x_lb=np.array([[20.0, 15.0,  0.0,     0.0]]),
                       x_ub = np.array([[23.0, 25.0, 50.0, 20000.0]])):
    x0 = x_lb + (X[0:4] * (x_ub - x_lb))[0]
    x0 = x0.reshape(4, 1)
    return x0


def scale_control_actions(u,
                          u_lb=np.array([[-1000, -1000]]),
                          u_ub = np.array([[1000, 1000]])):
    u = (u-u_lb)/(u_ub-u_lb)
    return u


def get_mpc_action(X):
    T_SR = extract_T_sr(X)
    model = template_model()
    mpc = template_mpc_shap(model, T_SR)
    simulator = template_simulator_shap(model, T_SR)
    x0 = extract_state_vars(X)
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()
    u0 = mpc.make_step(x0)
    if u0[0] > u0[1]:
        return scale_control_actions(np.array([u0[0][0], u0[-1][0]]))[0]
    else:
        return scale_control_actions(np.array([-u0[1][0], u0[-1][0]]))[0]
