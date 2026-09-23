import numpy as np
import taichi as ti

from src.consititutive_model.infinitesimal_strain.VonMisesSoftening import *
from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO


class VonMisesSoftening(ConstitutiveModelBase):
    def __init__(self, sims: Simulation):
        super().__init__()
        self.is_elastic = False
        self.add_material(
            sims.max_material_num,
            sims.material_type,
            sims.contact_detection,
            VonMisesSofteningModel,
        )
        if sims.configuration == "ULMPM":
            self.stateVars = ULStateVariable.field(shape=sims.max_particle_num)
        elif sims.configuration == "TLMPM":
            self.stateVars = TLStateVariable.field(shape=sims.max_particle_num)

        if sims.solver_type == "Implicit":
            self.stiffness_matrix = ti.Matrix.field(6, 6, float, shape=sims.max_particle_num)

    def get_state_vars_dict(self, start_particle, end_particle):
        epstrain = np.ascontiguousarray(self.stateVars.epstrain.to_numpy()[start_particle:end_particle])
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        return {"epstrain": epstrain, "estress": estress}

    def reload_state_variables(self, state_vars):
        state = state_vars.item() if hasattr(state_vars, "item") else state_vars
        estress = np.asarray(state["estress"], dtype=np.float64)
        epstrain = np.asarray(state["epstrain"], dtype=np.float64)
        kernel_reload_state_variables(estress, epstrain, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, "MaterialID")
        self.check_materialID(materialID, self.matProps.shape[0])

        if self.matProps[materialID].density > 0.0:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, "Density", 2650)
        young = DictIO.GetEssential(material, "YoungModulus")
        possion = DictIO.GetAlternative(material, "PossionRatio", 0.3)
        yield_peak = DictIO.GetEssential(material, "YieldStress")
        yield_residual = DictIO.GetAlternative(material, "ResidualYieldStress", yield_peak)
        epstrain_start = DictIO.GetAlternative(material, "PlasticDevStrain", 0.0)
        epstrain_end = DictIO.GetAlternative(material, "ResidualPlasticDevStrain", epstrain_start)

        if yield_peak < 0.0 or yield_residual < 0.0:
            raise ValueError("VonMisesSoftening yield stresses must be non-negative")
        if yield_residual > yield_peak:
            raise ValueError("ResidualYieldStress must be <= YieldStress for softening")
        if epstrain_end < epstrain_start:
            raise ValueError("ResidualPlasticDevStrain must be >= PlasticDevStrain")

        self.matProps[materialID].add_material(
            density,
            young,
            possion,
            yield_peak,
            yield_residual,
            epstrain_start,
            epstrain_end,
        )
        self.contact_initialize(material)
        self.matProps[materialID].print_message(materialID)

    def get_lateral_coefficient(self, materialID):
        mu = self.matProps[materialID].possion
        return mu / (1.0 - mu)
