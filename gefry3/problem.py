import numpy as np
import yaml
from gefry3.classes import *
from gefry3.classes.meta import Dictable
from copy import deepcopy

import warnings

# from abc import *

__all__ = [
    "SimpleProblem",
    "PerturbableXSProblem",
    "read_input_problem",
    "read_input",
    "write_input",
    "load_dict",
    "dump_dict"
]

class AmbiguousProblemSelectionError(Exception): pass

class BaseProblem(Dictable):
    @classmethod
    def get_loader(cls, name):
        # loader = [i for i in cls.__subclasses__() if i.PROBLEM_TYPE == name]

        # if len(loader) != 1:
            # # This really shouldn't happen unless something is really screwed up
            # raise AmbiguousProblemSelectionError(name)
        # else:
            # return loader[0] 

        return classRegistry[name]



class SimpleProblem(BaseProblem):
    PROBLEM_TYPE = "Simple_Problem"
    HAS_REFERENCES = True

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
            # dr = np.linalg.norm(np.asarray(detector.R) - np.asarray(r))
            
            # paths = self.domain.construct_path(r, detector.R)
            # alpha = np.exp(-(paths * self.Sigma_T).sum())

            # responses[i] = detector.compute_response(I * alpha / (4. * np.pi * (dr ** 2))) 

            responses[i] = self.compute_single_response(detector, r, I)

        return responses.astype(np.float64)

    def compute_jacobian(self, r, I):
        return np.array([self.compute_single_jacobian(d, r, I) for d in self.detectors])

    def compute_single_response(self, detector, r, I):
        #dr = np.linalg.norm(np.asarray(detector.R) - np.asarray(r))

        paths = self.domain.construct_path(r, detector.R)
        alpha = np.exp(-(paths * self.Sigma_T).sum())

        #response = detector.compute_response(I * alpha / (4. * np.pi * (dr ** 2.)))
        response = detector.compute_response(I * alpha, r)

        return response.astype(np.float64)

    def compute_single_jacobian(self, detector, r, I):
        paths = self.domain.construct_path(r, detector.R) # Memoize this?
        alpha = np.exp(-(paths * self.Sigma_T)) 

        d = self.compute_single_response(detector, r, I)

        return d * alpha


    def _as_dict(self):
        return {
            "domain": self.domain._as_dict(),
            "interstitial_material": self.interstitial_material._as_dict(),
            "materials": [i._as_dict() for i in self.materials],
            "source": self.source._as_dict(),
            "detectors": [i._as_dict() for i in self.detectors],
        }

    @classmethod
    def _from_dict(cls, data):
        return cls(
            Domain._from_dict(data["domain"]),
            Material._from_dict(data["interstitial_material"]),
            [Material._from_dict(i) for i in data["materials"]],
            Source._from_dict(data["source"]),
            [Detector._from_dict(i) for i in data["detectors"]],
        )


class PerturbableXSProblem(SimpleProblem):
    PROBLEM_TYPE = "Perturbable_XS_Problem"
    HAS_REFERENCES = False

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

def resolve_references(data_orig):
    data = deepcopy(data_orig)

    materials = data["materials"]
    material_refs = [i["material"] for i in data["domain"]["solids"]]

    # Delete the extraneous material references
    for solid in data["domain"]["solids"]:
        del solid["material"]

    materials_flattened = [materials[i] for i in material_refs]
    data["materials"] = materials_flattened

    return data

def compact_references(data_orig):
    data = deepcopy(data_orig)
    materials = [Material._from_dict(i) for i in data["materials"]]

    uniq_materials = []
    for m in materials:
        if m not in uniq_materials:
            uniq_materials.append(m)

    for i, material in enumerate(data["materials"]):
        m = Material._from_dict(material)
        ref = uniq_materials.index(m)
        data["domain"]["solids"][i]["material"] = ref

    data["materials"] = {i: material._as_dict() for i, material in enumerate(uniq_materials)}

    return data

def load_dict(data):
    problem_type = data["problem_type"]

    loader = BaseProblem.get_loader(problem_type)
    spec = resolve_references(data["data"])

    return loader._from_dict(spec) 

def dump_dict(problem):
    problem_type = problem.PROBLEM_TYPE

    return {
        "problem_type": problem_type,
        "data": compact_references(problem._as_dict()) if problem.HAS_REFERENCES else problem.as_dict(),
    }

# This function is deprecated! Originally you specified the type of problem
# in the input deck, but I've changed it to not need this.
def read_input(fname):
    warnings.warn("Old style input loading is deprecated", DeprecationWarning)
    with open(fname, 'r') as f:
        return load_dict(yaml.safe_load(f.read()))

def read_input_problem(fname, problem_type=None):
    with open(fname, 'r') as f:
        data = yaml.safe_load(f.read())

    if problem_type is not None:
        if "problem_type" in data and data["problem_type"] is not None:
            ws = "Input specifies a problem type [{}], but user specified [{}] - the type in the input file will be overidden." \
                .format(data["problem_type"], problem_type)

            warnings.warn(ws)

        data["problem_type"] = problem_type

    elif "problem_type" not in data: # none in input, none provided
        warnings.warn("No problem type provided and none in input file, defaulting to Simple_Problem")
        data["problem_type"] = "Simple_Problem"

    return load_dict(data)

def write_input(fname, problem):
    with open(fname, 'w') as f:
        f.write(yaml.safe_dump(dump_dict(problem))) 

classRegistry = {
    "Simple_Problem": SimpleProblem,
    "Perturbable_XS_Problem": PerturbableXSProblem,
} 
