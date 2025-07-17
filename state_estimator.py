import numpy as np

def kalman_filter(x, u, s_x, y, A, B, C, s_w, s_e):
    '''
    Kalman filter to update the posterior of the control state given subgaussian noise
    x: last est state
    u: action
    s_x: current variance
    y: current observation
    A, B: dynamics
    C: measurement model
    K: control gain
    s_w: disturbance variance
    s_e: measurement variance
    '''
    # update variance mat
    s_x_prior = A @ s_x @ A.T + s_w
    s_y_prior = C @ s_x_prior @ C.T + s_e
    s_y_prior_inv = np.linalg.inv(s_y_prior)
    state_obs_cov = s_x_prior @ C.T

    # compute the kalman gain
    L = state_obs_cov @ s_y_prior_inv

    # get prediction
    x_prior = A @ x + B @ u
    y_prior = C @ x_prior

    # update the posterior
    post_mean = x_prior + L @ (y - y_prior)
    post_var = s_x_prior - state_obs_cov @ s_y_prior_inv @ state_obs_cov.T

    return post_mean, post_var


def kalman_filter_gain_propagation(s_x, A, C, s_w, s_e):
    # update variance mat
    s_x_prior = A @ s_x @ A.T + s_w
    s_y_prior = C @ s_x_prior @ C.T + s_e
    s_y_prior_inv = np.linalg.inv(s_y_prior)
    state_obs_cov = s_x_prior @ C.T

    L = state_obs_cov @ s_y_prior_inv

    post_var = s_x_prior - state_obs_cov @ s_y_prior_inv @ state_obs_cov.T

    return post_var, L


def kalman_filter_gain_rollout(s_x, A, C, s_w, s_e, T):

    s_x_prior = s_x
    
    s_y_prior = C @ s_x_prior @ C.T + s_e
    s_y_prior_inv = np.linalg.inv(s_y_prior)
    state_obs_cov = s_x_prior @ C.T
    L = state_obs_cov @ s_y_prior_inv

    post_var = s_x_prior - state_obs_cov @ s_y_prior_inv @ state_obs_cov.T

    Ls = [L]

    for t in range(T):
        post_var, L = kalman_filter_gain_propagation(post_var, A, C, s_w, s_e)
        Ls.append(L)

    return Ls


def kalman_filter_fix_gain(x, u, s_x, y, A, B, C, L, s_w, s_e):
    '''
    Kalman filter to update the posterior of the control state given subgaussian noise
    x: last est state
    u: action
    s_x: current variance
    y: current observation
    A, B: dynamics
    C: measurement model
    K: control gain
    s_w: disturbance variance
    s_e: measurement variance
    '''
    # update variance mat
    s_x_prior = A @ s_x @ A.T + s_w
    s_y_prior = C @ s_x_prior @ C.T + s_e
    s_y_prior_inv = np.linalg.inv(s_y_prior)
    state_obs_cov = s_x_prior @ C.T

    # get prediction
    x_prior = A @ x + B @ u
    y_prior = C @ x_prior

    # update the posterior
    post_mean = x_prior + L @ (y - y_prior)
    post_var = s_x_prior - state_obs_cov @ s_y_prior_inv @ state_obs_cov.T

    return post_mean, post_var