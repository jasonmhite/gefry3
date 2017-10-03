import gefry3
import pymc

import numpy as np

# Load problem and set up some parameters

P = gefry3.read_input_problem(
    'g3_deck.yml',
    problem_type="Simple_Problem",
)

NS = int(1e6) # Number of samples to draw
S0 = P.source.R # m
I0 = P.source.I0 # Bq
BG = 300 # cps

ND = len(P.detectors)

XMIN, YMIN, XMAX, YMAX = P.domain.all.bounds
IMIN, IMAX = I0 * 0.1, I0 * 10

# Generate the synthetic measurement data

DWELL = np.array([i.dwell for i in P.detectors])

# Call P at the true values to get the real response
nominal = P(S0, I0)
data = np.random.poisson(nominal + BG * DWELL) 

# NOTE that if you want to compare different runs then you need to
# save these to a file and use the same data, obviously.

def model_factory():
    """Build a PyMC model and return it as a dict"""

    x = pymc.Uniform("x", value=S0[0], lower=XMIN, upper=XMAX)
    y = pymc.Uniform("y", value=S0[1], lower=YMIN, upper=YMAX)
    I = pymc.Uniform("I", value=I0, lower=IMIN, upper=IMAX)

    @pymc.deterministic(plot=False)
    def model_pred(x=x, y=y, I=I):
        return P([x, y], I)

    detector_response = pymc.Poisson(
        "d",
        data,
        value=data,
        observed=True,
        plot=False,
    )

    background = pymc.Poisson(
        "background",
        DWELL * BG,
        value=DWELL * BG,
        observed=True,
        plot=False,
    )

    observed_response = model_pred + background

    # return locals() # the lazy way

    return {
        "x": x,
        "y": y,
        "I": I,
        "detector_response": detector_response,
        "background": background,
        "observed_response": observed_response,
    }

# Instantiate the sampler and run

mvars = model_factory()
M = pymc.MCMC(mvars)

# It's safe to set a sampler on all the variables, even though only
# some will actually use it.
M.use_step_method(
    pymc.AdaptiveMetropolis,
    [mvars[i] for i in mvars]
)

M.sample(NS)

# Save the chains to "out.dat"
res = np.vstack([M.trace(z)[:] for z in ["x", "y", "I"]])
np.savetxt("out.dat", res.T)

# Print the means for a quick check
print("\n\n==== Results ====\n")
print("x: {}".format(np.mean(M.trace("x")[:])))
print("y: {}".format(np.mean(M.trace("y")[:])))
print("I: {}".format(np.mean(M.trace("I")[:])))
