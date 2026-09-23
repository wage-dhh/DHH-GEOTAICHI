import taichi as ti

from src.consititutive_model.MaterialKernel import *
from src.utils.constants import DELTA2D, DELTA, FTOL, MAXITS, Threshold
from src.utils.MatrixFunction import matrix_form
from src.utils.TypeDefination import mat3x3
from src.utils.VectorFunction import voigt_form


@ti.dataclass
class ULStateVariable:
    epstrain: float
    estress: float

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        self.epstrain = 0.0
        self.estress = EquivalentStress(particle[np].stress)

    @ti.func
    def _update_vars(self, stress, epstrain):
        self.estress = EquivalentStress(stress)
        self.epstrain = epstrain


@ti.dataclass
class TLStateVariable:
    epstrain: float
    estress: float
    deformation_gradient: mat3x3
    stress: mat3x3

    @ti.func
    def _initialize_vars(self, np, particle, matProps):
        self.epstrain = 0.0
        self.estress = EquivalentStress(particle[np].stress)
        self.deformation_gradient = DELTA
        self.stress = matrix_form(particle[np].stress)

    @ti.func
    def _update_deformation_gradient(self, deformation_gradient_rate, dt):
        self.deformation_gradient += deformation_gradient_rate * dt[None]

    @ti.func
    def _update_vars(self, stress, epstrain):
        self.estress = EquivalentStress(stress)
        self.epstrain = epstrain


@ti.dataclass
class VonMisesSofteningModel:
    density: float
    young: float
    possion: float
    shear: float
    bulk: float
    yield_peak: float
    yield_residual: float
    epstrain_start: float
    epstrain_end: float

    def add_material(
        self,
        density,
        young,
        possion,
        yield_peak,
        yield_residual,
        epstrain_start,
        epstrain_end,
    ):
        self.density = density
        self.young = young
        self.possion = possion
        self.shear = 0.5 * self.young / (1.0 + self.possion)
        self.bulk = self.young / (3.0 * (1.0 - 2.0 * self.possion))
        self.yield_peak = yield_peak
        self.yield_residual = yield_residual
        self.epstrain_start = epstrain_start
        self.epstrain_end = epstrain_end

    def add_contact_parameter(self, friction, kn, kt):
        self.friction = friction
        self.kn = kn
        self.kt = kt

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model: von Mises isotropic softening model')
        print("Model ID: ", materialID)
        print('Density: ', self.density)
        print('Young Modulus: ', self.young)
        print('Possion Ratio: ', self.possion)
        print('Yield Stress (peak): ', self.yield_peak)
        print('Yield Stress (residual): ', self.yield_residual)
        print('Plastic Strain Softening Interval: ', self.epstrain_start, self.epstrain_end, '\n')

    @ti.func
    def _get_sound_speed(self):
        sound_speed = 0.0
        if self.density > 0.0:
            sound_speed = ti.sqrt(
                self.young
                * (1.0 - self.possion)
                / (1.0 + self.possion)
                / (1.0 - 2.0 * self.possion)
                / self.density
            )
        return sound_speed

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        return (DELTA + velocity_gradient * dt[None]).determinant()

    @ti.func
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        return (DELTA2D + velocity_gradient * dt[None]).determinant()

    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        return 1.0 + dt[None] * (strain_rate[0] + strain_rate[1] + strain_rate[2])

    @ti.func
    def PK2CauchyStress(self, np, stateVars):
        inv_j = 1.0 / stateVars[np].deformation_gradient.determinant()
        return voigt_form(stateVars[np].stress @ stateVars[np].deformation_gradient.transpose() * inv_j)

    @ti.func
    def Cauchy2PKStress(self, np, stateVars, stress):
        j = stateVars[np].deformation_gradient.determinant()
        return matrix_form(stress) @ stateVars[np].deformation_gradient.inverse().transpose() * j

    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):
        de = calculate_strain_increment2D(velocity_gradient, dt)
        dw = calculate_vorticity_increment2D(velocity_gradient, dt)
        return self.core_2d(np, previous_stress, de, dw, stateVars)

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):
        de = calculate_strain_increment(velocity_gradient, dt)
        dw = calculate_vorticity_increment(velocity_gradient, dt)
        return self.core(np, previous_stress, de, dw, stateVars)

    @ti.func
    def rotate_stress_hughes_winget(self, stress, spin_increment):
        spin = ti.Matrix(
            [
                [0.0, -spin_increment[0], spin_increment[2]],
                [spin_increment[0], 0.0, -spin_increment[1]],
                [-spin_increment[2], spin_increment[1], 0.0],
            ]
        )
        identity = ti.Matrix.identity(float, 3)
        # Analytic Cayley transform for a 3D skew matrix:
        # (I-W/2)^-1(I+W/2) = I + 4W/(4+w.w) + 2W^2/(4+w.w).
        # This is algebraically identical to Hughes-Winget Eq. (28), while
        # avoiding a 3x3 inverse for every material point at every step.
        spin_norm_squared = spin_increment.dot(spin_increment)
        cayley_factor = 4.0 / (4.0 + spin_norm_squared)
        rotation = identity + cayley_factor * spin + 0.5 * cayley_factor * (spin @ spin)
        stress_matrix = matrix_form(stress)
        return voigt_form(rotation @ stress_matrix @ rotation.transpose())

    @ti.func
    def rotate_stress_hughes_winget_2d(self, stress, spin_increment):
        half_spin = 0.5 * spin_increment[0]
        denominator = 1.0 + half_spin * half_spin
        cosine = (1.0 - half_spin * half_spin) / denominator
        sine = 2.0 * half_spin / denominator
        cosine_squared = cosine * cosine
        sine_squared = sine * sine
        sine_cosine = sine * cosine
        return vec6f(
            cosine_squared * stress[0] + sine_squared * stress[1] - 2.0 * sine_cosine * stress[3],
            sine_squared * stress[0] + cosine_squared * stress[1] + 2.0 * sine_cosine * stress[3],
            stress[2],
            sine_cosine * (stress[0] - stress[1]) + (cosine_squared - sine_squared) * stress[3],
            sine * stress[5] + cosine * stress[4],
            cosine * stress[5] - sine * stress[4],
        )

    @ti.func
    def yield_strength(self, epstrain):
        sigma_y = self.yield_peak
        if self.epstrain_end <= self.epstrain_start + Threshold:
            if epstrain > self.epstrain_start:
                sigma_y = self.yield_residual
        elif epstrain >= self.epstrain_end:
            sigma_y = self.yield_residual
        elif epstrain > self.epstrain_start:
            ratio = (epstrain - self.epstrain_start) / (self.epstrain_end - self.epstrain_start)
            sigma_y = self.yield_peak + ratio * (self.yield_residual - self.yield_peak)
        return sigma_y

    @ti.func
    def yield_slope(self, epstrain):
        slope = 0.0
        if epstrain > self.epstrain_start and epstrain < self.epstrain_end:
            interval = self.epstrain_end - self.epstrain_start
            if ti.abs(interval) > Threshold:
                slope = (self.yield_residual - self.yield_peak) / interval
        return slope

    @ti.func
    def ComputeYieldFunction(self, stress, epstrain):
        return EquivalentStress(stress) - self.yield_strength(epstrain)

    @ti.func
    def ComputeYieldState(self, stress, epstrain):
        f_function = self.ComputeYieldFunction(stress, epstrain)
        yield_state = 0
        if f_function > -1.0e-8:
            yield_state = 1
        return yield_state, f_function

    @ti.func
    def ComputeDfDsigma(self, stress):
        return DqDsigma(stress)

    @ti.func
    def ComputeDgDsigma(self, stress):
        dg_dp = 0.0
        dg_dq = 1.0
        dp_dsigma = DpDsigma()
        dq_dsigma = DqDsigma(stress)
        dg_dsigma = dg_dp * dp_dsigma + dg_dq * dq_dsigma
        return dg_dp, dg_dq, dg_dsigma

    @ti.func
    def ComputeElasticStress(self, dstrain, stress):
        return stress + self.ComputeElasticStressIncrement(dstrain)

    @ti.func
    def ComputeElasticStressIncrement(self, dstrain):
        return ElasticTensorMultiplyVector(dstrain, self.bulk, self.shear)

    @ti.func
    def radial_return(self, trial_stress, epstrain):
        q_trial = EquivalentStress(trial_stress)
        yield_old = self.yield_strength(epstrain)
        dlambda = 0.0
        if q_trial > yield_old + FTOL:
            denominator = 3.0 * self.shear
            initial_denominator = ti.max(denominator, Threshold)
            dlambda = ti.max(0.0, (q_trial - yield_old) / initial_denominator)

            # Solve q_trial - 3G*dlambda = sigma_y(epstrain + dlambda).
            # The fixed iteration handles the peak, linear-softening, and residual branches.
            for _ in range(MAXITS):
                ep_new = epstrain + dlambda
                yield_new = self.yield_strength(ep_new)
                softening_slope = self.yield_slope(ep_new)
                denominator = 3.0 * self.shear + softening_slope
                candidate = dlambda
                if ti.abs(denominator) > Threshold:
                    residual = q_trial - 3.0 * self.shear * dlambda - yield_new
                    candidate = ti.max(0.0, dlambda + residual / denominator)
                if ti.abs(candidate - dlambda) <= FTOL:
                    dlambda = candidate
                    break
                dlambda = candidate

        q_new = ti.max(0.0, q_trial - 3.0 * self.shear * dlambda)
        ratio = q_new / q_trial if q_trial > Threshold else 1.0
        mean_stress = MeanStress(trial_stress)
        deviatoric_stress = DeviatoricStress(trial_stress)
        updated_stress = vec6f(
            mean_stress + ratio * deviatoric_stress[0],
            mean_stress + ratio * deviatoric_stress[1],
            mean_stress + ratio * deviatoric_stress[2],
            ratio * deviatoric_stress[3],
            ratio * deviatoric_stress[4],
            ratio * deviatoric_stress[5],
        )
        return updated_stress, dlambda

    @ti.func
    def core(self, np, previous_stress, de, dw, stateVars):
        rotated_stress = self.rotate_stress_hughes_winget(previous_stress, dw)
        return self.integrate_rotated_stress(np, rotated_stress, de, stateVars)

    @ti.func
    def core_2d(self, np, previous_stress, de, dw, stateVars):
        rotated_stress = self.rotate_stress_hughes_winget_2d(previous_stress, dw)
        return self.integrate_rotated_stress(np, rotated_stress, de, stateVars)

    @ti.func
    def integrate_rotated_stress(self, np, rotated_stress, de, stateVars):
        trial_stress = rotated_stress + self.ComputeElasticStressIncrement(de)
        updated_stress, pdstrain = self.radial_return(trial_stress, stateVars[np].epstrain)
        stateVars[np].epstrain += pdstrain
        stateVars[np].estress = EquivalentStress(updated_stress)
        return updated_stress

    @ti.func
    def ComputePKStress(self, np, velocity_gradient, stateVars, dt):
        previous_stress = self.PK2CauchyStress(np, stateVars)
        cauchy_stress = self.ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
        PKstress = self.Cauchy2PKStress(np, stateVars, cauchy_stress)
        stateVars[np].stress = PKstress
        return PKstress

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        return ComputeElasticStiffnessTensor(self.bulk, self.shear)

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stateVars):
        return ComputeElasticStiffnessTensor(self.bulk, self.shear)


@ti.kernel
def kernel_reload_state_variables(estress: ti.types.ndarray(), epstrain: ti.types.ndarray(), state_vars: ti.template()):
    for np in range(estress.shape[0]):
        state_vars[np].estress = estress[np]
        state_vars[np].epstrain = epstrain[np]
