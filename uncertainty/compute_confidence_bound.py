
import jax.numpy as jnp
import numpy as np
from scipy import stats

def tail_bound_from_vp(t, n):
    '''
    t: norm of variable
    n: dimension of variable
    '''

    return 2*jnp.exp(-(1/2 - 1/2**(1 + 2/n))*t**2)

def tail_bound_from_vp_and_m(t, m, n):
    '''
    t: norm of variable
    m: optimized positive value
    n: dimension of variable
    '''

    return (
        ((1 + m) / m) ** (n / 2)
        * jnp.exp(
            - t**2 
            / (2 + 2 * m)
        )
    )


def tail_bound_from_C4(t):
    '''
    t: norm of variable
    n: dimension of variable
    '''

    return 2*np.exp(-t**2)


def get_bound_given_probability_from_vp(prob, vp, n):
    '''
    prob: small prob of outside the bound
    vp: variance proxy
    n: dimension
    '''
    t = np.sqrt(2/(1 - 1/2**(2/n)) * np.log(2/prob)) * vp
    return t

def get_bound_given_probability_from_vp_and_m(prob, vp, m, n):
    '''
    prob: small prob of outside the bound
    vp: variance proxy
    m: positive constant
    n: dimension
    '''
    return np.sqrt(
            (1 + m)
            * (n 
                * jnp.log((1 + m) / m)
                + 2
                * jnp.log(1 / prob)
            )
        )*vp

def get_bound_scale_given_probability_from_m(prob, m, n):
    '''
    prob: small prob of outside the bound
    vp: variance proxy
    m: positive constant
    n: dimension
    '''
    return np.sqrt(
            (1 + m)
            * (n 
                * jnp.log((1 + m) / m)
                + 2
                * jnp.log(1 / prob)
            )
        )


def get_bound_given_probability_from_c4(prob, c4):
    '''
    prob: small prob of outside the bound
    vp: variance proxy
    n: dimension
    '''
    t = np.sqrt(np.log(2/prob)) * c4
    return t


def tail_bound_density_from_c4(x_1d, c4):
    '''
    get derivative of the density of the tail bound given a norm
    x_1d: 1 dimensional projection of x (can be signed norm)
    '''
    results = np.zeros(x_1d.shape)
    results[x_1d < 0] = -2 * x_1d[x_1d < 0] / c4**2 * np.exp(-x_1d[x_1d < 0]**2 / c4**2)
    results[x_1d >= 0] = 2 * x_1d[x_1d >= 0] / c4**2 * np.exp(-x_1d[x_1d >= 0]**2 / c4**2)

    return results



def tail_bound_density_from_vp(x_1d, n, vp):
    '''
    get derivative of the density of the tail bound given a norm
    x_1d: 1 dimensional projection of x (can be signed norm)
    n: number of dimension
    '''
    a = (1/2 - 1/2**(1 + 2/n))

    results = np.zeros(x_1d.shape)

    results[x_1d < 0] =  -2 * a * x_1d[x_1d < 0] / vp**2 * np.exp(-a*x_1d[x_1d < 0]**2 / vp**2)
    results[x_1d >= 0] = 2 * a * x_1d[x_1d >= 0] / vp**2 * np.exp(-a*x_1d[x_1d >= 0]**2 / vp**2)

    return results


def conformal_prediction_state(true_states, nominal_state, prob):
    '''
    true_states: (N, d) array of true states
    norminal_state: (d,) array of norminal state
    prob: small prob of outside the bound
    '''
    # first pca
    diffs = true_states - nominal_state
    cov = diffs.T @ diffs / diffs.shape[0] # (d, d)
    # cov = np.eye(cov.shape[0]) # TODO: change back to no covariance
    eigvals, eigvecs = np.linalg.eig(cov)

    # then get norm
    norms = [diffs[i].reshape((1, -1)) @ np.linalg.inv(cov + 1e-10) @ diffs[i].reshape((-1, 1)) for i in range(diffs.shape[0])]
    norms = np.array(norms).reshape((-1,))
    # sort the norms
    sort_index = np.argsort(norms)
    
    # get the index
    n = norms.shape[0]
    m = min(np.floor((n + 1) * (1 - prob)), n-1)

    return norms[sort_index[int(m)]] * cov

def conformal_predicton_trajectory(all_xs, all_zs, num_steps, prob, last_dim=None):
    # conformal prediction
    all_xs_array = np.asarray(all_xs)
    all_zs_array = np.asarray(all_zs)
    if last_dim is not None:
        all_xs_array = all_xs_array[:, :, :last_dim]
        all_zs_array = all_zs_array[:, :, :last_dim]
    cp_bounds = []
    for t in range(num_steps+1):
        # get the true state
        true_state = all_xs_array[:, t, :]
        # get the nominal state
        nominal_state = all_zs_array[:, t, :]
        bound = conformal_prediction_state(true_states=true_state, nominal_state=nominal_state, prob=prob)
        cp_bounds.append(bound)

    return cp_bounds



def gaussian_scale_from_prob(prob, dim, range, num):
    '''
    prob: small prob of outside the bound
    range: range of search
    num: number of search
    '''
    xs = np.linspace(range[0], range[1], num)

    cdfs = 1 - stats.chi.cdf(xs, dim)

    index = np.argmin(np.abs(cdfs - prob))

    return xs[index]