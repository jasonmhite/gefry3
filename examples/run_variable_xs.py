import gefry3 # My model code
import pymc

import numpy as np

from scipy.stats import multivariate_normal 

P = gefry3.read_input_problem(
    'g3_deck.yml',
    problem_type="Perturbable_XS_Problem",
)

NS = int(1e6) # Number of samples

S0 = P.source.R # m
I0 = P.source.I0 # Bq
BG = 300 # cps

XMIN, YMIN, XMAX, YMAX = P.domain.all.bounds
IMIN, IMAX = 1e9, 5e9

# Relative perturbation used for all cross sections
XS_DELTA = 0.50

# Generate some data

DWELL = np.array([i.dwell for i in P.detectors])

# Call P at the nominal values to get the real response
nominal = P(
    S0,
    I0,
    P.interstitial_material,
    P.materials,
)

nominal += BG * DWELL

# Generate the data and the covariance assuming detectors are independent
# (a pretty safe assumption).
data = np.random.poisson(nominal)
C_data = np.diag(data) 

def model_factory():
    """Build a PyMC model and return it as a dict"""

    x = pymc.Uniform("x", value=S0[0], lower=XMIN, upper=XMAX)
    y = pymc.Uniform("y", value=S0[1], lower=YMIN, upper=YMAX)
    I = pymc.Uniform("I", value=I0, lower=IMIN, upper=IMAX)

    # Distributions for the cross sections

    # Just the interstitial material
    s_i_xs = P.interstitial_material.Sigma_T
    interstitial_xs = pymc.Uniform(
        "Sigma_inter",
        s_i_xs * (1 - XS_DELTA),
        s_i_xs * (1 + XS_DELTA),
        value=s_i_xs,
        observed=True,
    )

    # All the rest
    mu_xs = np.array([M.Sigma_T for M in P.materials])

    building_xs = pymc.Uniform(
        "Sigma",
        mu_xs * (1 - XS_DELTA),
        mu_xs * (1 + XS_DELTA),
        value=mu_xs,
        observed=True,
    )

    # Predictions

    @pymc.deterministic(plot=False)
    def model_pred(x=x, y=y, I=I, interstitial_xs_p=interstitial_xs, building_xs_p=building_xs):
        # The _p annotation is so that I can access the actual stochastics
        # in the enclosing scope, see down a couple lines where I resample

        inter_mat = gefry3.Material(1.0, interstitial_xs_p)
        building_mats = [gefry3.Material(1.0, s) for s in building_xs_p]

        # Force the cross sections to be resampled
        interstitial_xs.set_value(interstitial_xs.random(), force=True)
        building_xs.set_value(building_xs.random(), force=True)

        return P(
            [x, y],
            I,
            inter_mat,
            building_mats,
        )

    background = pymc.Poisson(
        "b",
        DWELL * BG,
        value=DWELL * BG,
        observed=True,
        plot=False,
    ) 

    @pymc.stochastic(plot=False, observed=True)
    def observed_response(value=nominal, model_pred=model_pred, background=background):
        resp = model_pred + background

        return multivariate_normal.logpdf(resp, mean=data, cov=C_data)

    return {
        "x": x,
        "y": y,
        "I": I,
        "interstitial_xs": interstitial_xs,
        "building_xs": building_xs,
        "model_pred": model_pred,
        "background": background,
        "observed_response": observed_response,
    }  

# Set up the sampler and run
mvars = model_factory()

M = pymc.MCMC(mvars)

M.use_step_method(
    pymc.AdaptiveMetropolis,
    [mvars[i] for i in mvars]
)

M.sample(NS)

# Summary stats and save the data
print("\n\n==== Results ====\n")
print("x: {} [{}]".format(np.mean(M.trace("x")[:]), np.std(M.trace("x")[:])))
print("y: {} [{}]".format(np.mean(M.trace("y")[:]), np.std(M.trace("y")[:])))
print("I: {} [{}]".format(np.mean(M.trace("I")[:]), np.std(M.trace("I")[:])))

res = np.vstack([M.trace(z)[:] for z in ["x", "y", "I"]])
np.savetxt("out_{}.dat".format(int(100 * XS_DELTA)), res.T) 
