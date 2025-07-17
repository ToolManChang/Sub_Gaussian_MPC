from uncertainty.estimate_variance_proxy import *
from uncertainty.optimize_confidence_interval import *
from uncertainty.compute_confidence_bound import *
from uncertainty.variance_proxy_propagation import *
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import gamma
from scipy.optimize import fsolve


if __name__=="__main__":
    

    bounds_sG = []
    bounds_Gau = []
    ratio = []
    diff = []
    bounds_guess = []
    prob = 0.01

    for n in range(1, 100):
        m = optimize_confidence_interval(prob, n)
        bound_SG = get_bound_scale_given_probability_from_m(prob, m, n)
        bound_Gau = stats.chi.ppf(1 - prob, n)

        # sim prob
        sim_prob = (prob
                    / np.exp(n/2)
                    * n**(n/2)
                    / 2**(n/2+1)
                    / gamma(n/2+1))
        solution = fsolve(lambda x: stats.chi2.pdf(x, n + 2) - sim_prob, bound_Gau**2)
        guess = np.sqrt(solution)

        bounds_sG.append(bound_SG)
        bounds_Gau.append(bound_Gau)
        bounds_guess.append(guess)
        ratio.append(bound_SG/bound_Gau)
        diff.append(bound_SG-bound_Gau)


        print(f"n: {n}, m: {m}, bound_SG: {bound_SG}, bound_Gau: {bound_Gau}, guess: {guess}")

    plt.plot(bounds_sG, label='sG')
    plt.plot(bounds_Gau, label='Gau')
    plt.plot(bounds_guess, label='guess')
    plt.legend()
    plt.savefig('bounds.png')

    plt.clf()
    plt.plot(ratio, label='SG/Gau')
    plt.plot(diff, label='SG-Gau')
    plt.legend()
    plt.savefig('ratio_diff.png')
    
    