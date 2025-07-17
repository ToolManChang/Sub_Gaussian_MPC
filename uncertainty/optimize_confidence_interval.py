import jax
import jax.nn as jnn
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative
from scipy import stats
from jaxopt._src.proximal_gradient import ProxGradState


def optimize_confidence_interval(
    prob: float, n: int, tol: float = 1e-15
) -> tuple[float, ProxGradState]:
    """optimize the normalized confidence interval given a probability 

    Args:
        prob: a small tail probability

    Returns:
        float: size of the normalized confidence interval
    """

    def f(m):
        return (
            (1 + m)
            * (n 
                * jnp.log((1 + m) / m)
                + 2
                * jnp.log(1 / prob)
            )
        )[0]

    solver = ProjectedGradient(fun=f, projection=projection_non_negative, tol=tol)
    m, state = solver.run(2*jax.numpy.ones((1,)))
    return m