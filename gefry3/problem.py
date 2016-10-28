import numpy as np
import yaml
from gefry3.classes import *
from gefry3.classes.meta import Dictable
from copy import deepcopy

# from abc import *

__all__ = ["SimpleProblem", "PerturbableXSProblem", "read_input", "write_input", "load_dict", "dump_dict"]

class AmbiguousProblemSelectionError(Exception): pass

class BaseProblem(Dictable):
    @classmethod
    def get_loader(cls, name):
        loader = [i for i in cls.__subclasses__() if i.PROBLEM_TYPE == name]

        if len(loader) != 1:
            # This really shouldn't happen unless something is really screwed up
            raise AmbiguousProblemSelectionError(problem_type)
        else:
            return loader[0] 

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
            dr = np.linalg.norm(np.asarray(detector.R) - np.asarray(r))
            
            paths = self.domain.construct_path(r, detector.R)
            alpha = np.exp(-(paths * self.Sigma_T).sum())

            responses[i] = detector.compute_response(I * alpha / (4. * np.pi * (dr ** 2))) 

        return responses.astype(np.float64)

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


class PerturbableXSProblem(BaseProblem):
    PROBLEM_TYPE = "Pertubable_XS_Problem"
    HAS_REFERENCES = False

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

    def _as_dict(self):
        return {
            "domain": self.domain._as_dict(),
            "source": self.source._as_dict(),
            "detectors": [i._as_dict() for i in self.detectors],
        }

    @classmethod
    def _from_dict(cls, data):
        return cls(
            Domain._from_dict(data["domain"]),
            Source._from_dict(data["source"]),
            [Detector._from_dict(i) for i in data["detectors"]],
        ) 


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
    spec = resolve_references(data["data"]) if loader.HAS_REFERENCES else data["data"]

    return loader._from_dict(spec) 

def dump_dict(problem):
    problem_type = problem.PROBLEM_TYPE

    return {
        "problem_type": problem_type,
        "data": compact_references(problem._as_dict()) if problem.HAS_REFERENCES else problem.as_dict(),
    }

def read_input(fname):
    with open(fname, 'r') as f:
        return load_dict(yaml.safe_load(f.read()))

def write_input(fname, problem):
    with open(fname, 'w') as f:
        f.write(yaml.safe_dump(dump_dict(problem)))
