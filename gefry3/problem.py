import numpy as np
from gefry3.classes import *

__all__ = ["SimpleProblem", "PerturbableXSProblem"]

class SimpleProblem(object):
    # Single source, fixed materials
    def __init__(self, domain, interstitial_material, materials, source, detectors):
        self.domain = domain
        self.source = source
        self.interstitial_material = interstitial_material
        self.materials = materials
        self.detectors = detectors

        # cache sigmas
        self.Sigma_T = np.array(
            [self.interstitial_material.Sigma_T] \
                + [M.Sigma_T for M in self.materials]
        )

    def __call__(self, r, I):
        # Compute response to a source at (r,I)

        responses = np.zeros_like(self.detectors)

        for (i, detector) in enumerate(self.detectors):
            # Compute attenuation
            dr = np.linalg.norm(np.asarray(detector.R) - np.asarray(r))
            
            paths = self.domain.construct_path(r, detector.R)
            alpha = np.exp(-(paths * self.Sigma_T).sum())

            responses[i] = (1 / (4. * np.pi * (dr ** 2))) \
                    * alpha \
                    * detector.compute_response(I)

        return responses.astype(np.float64)

class PerturbableXSProblem(object):
    def __init__(self, domain, source, detectors):
        self.domain = domain
        self.source = source
        self.detectors = detectors

    def __call__(self, r, I, interstitial_material, materials):
        # MATERIALS MUST BE IN SAME ORDER AS SOLIDS

        # Allocate a temporary fixed problem and evaluate
        return SimpleProblem(
            self.domain,
            interstitial_material,
            materials,
            self.source,
            self.detectors
        )(r, I)
