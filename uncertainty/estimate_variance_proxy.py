import jax
import jax.nn as jnn
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
from jaxopt._src.lbfgs import LbfgsState
from scipy import stats
import skgeom as sg

jax.config.update("jax_enable_x64", True)



def mfg_bound(sample, c4):
    return jnp.mean(jnp.exp((sample / c4) ** 2))


def find_C4(samples, search_range, search_int):
    c4_values = np.linspace(search_range[0], search_range[1], search_int)

    bounds = np.asarray([mfg_bound(samples, c4) for c4 in c4_values]) - 2.0
    # Using the fact that `mfg_bound` is monotonically decreasing w.r.t c4, find the first c4
    # value that causes bounds to be non-negative
    best_value = c4_values[jnp.argmin(bounds > 0)]

    return best_value


def estimate_variance_proxy(
    sample: jax.Array, tol: float = 1e-20
) -> tuple[float, LbfgsState]:
    """Estimates the variance proxy of a sub-Gaussian distribution

    Args:
        sample (jax.Array): A 2D array of shape (n_samples, n_features)

    Returns:
        float: An estimate of the variance proxy
    """
    if sample.ndim != 2:
        raise ValueError("Sample must be a 2D array")
    
    mu = sample.mean(0)
    debiased_sample = sample - mu

    def f(lambda_):
        return (
            -2.0
            * (
                jnn.logsumexp(jnp.tensordot(debiased_sample, lambda_, axes=1))
                - jnp.log(sample.shape[0])
            )
            / (lambda_.dot(lambda_))
        )

    solver = jaxopt.LBFGS(fun=f, tol=tol)
    params, state = solver.run(1.0*jax.numpy.ones((sample.shape[1],)))
    return jax.numpy.sqrt(-state.value), params


def compute_bound_box(sample):
    bounds = jnp.max(jnp.abs(sample), axis=0)
    poly = sg.Polygon([
        sg.Point2(bounds[0], bounds[1]), 
        sg.Point2(-bounds[0], bounds[1]),
        sg.Point2(-bounds[0], -bounds[1]), 
        sg.Point2(bounds[0], -bounds[1]), 
    ])
    return poly, bounds


def compute_bound_points(sample):
    max_points = np.max(sample, axis=0)
    min_points = np.min(sample, axis=0)
    points = np.zeros((2**sample.shape[1], sample.shape[1]))
    for i in range(2**sample.shape[1]):
        for j in range(sample.shape[1]):
            if i & (1 << j):
                points[i, j] = max_points[j]
            else:
                points[i, j] = min_points[j]

    return points


def compute_gaussian_bound(sample):
    std = jnp.std(sample, axis=0)
    return 3 * std


def estimate_scaled_covariance_proxy(sample):
    cov = sample.T @ sample  # (d, d)
    L = np.linalg.cholesky(cov)
    n_samples = np.dot(sample, np.linalg.inv(L).T)
    vp, _ = estimate_variance_proxy(n_samples)
    return L @ L.T * vp**2
