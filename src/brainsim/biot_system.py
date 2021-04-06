import fenics as pde
from fenics import (
    dx, inner, grad, div
)


def epsilon(u):
    return pde.grad(u)


class BiotSystem:
    def __init__(
        self,
        mesh,
        shear_modulus,
        lame_parameter,
        biot_coefficient,
        hydraulic_conductivity,
        force_field,
        source_field,
        timestep,
    ):
        """Note: All parameters should be scaled by the average shear modulus.
        """
        self.mesh = mesh
        self.function_space = self.make_function_space()
        self.previous_solution = pde.Function(self.function_space)
        self.solution = pde.Function(self.function_space)
        self.shear_modulus = shear_modulus
        self.lame_parameter = lame_parameter
        self.biot_coefficient = biot_coefficient
        self.hydraulic_conductivity = hydraulic_conductivity
        self.force_field = force_field
        self.source_field = source_field
        self.timestep = timestep

    def make_function_space(self):
        P2vec = pde.VectorElement('CG', self.mesh.ufl_cell(), 2)
        P1 = pde.FiniteElement('CG', self.mesh.ufl_cell(), 1)

        return pde.FunctionSpace(self.mesh, pde.MixedElement([P2vec, P1, P1]))

    @property
    def trial_functions(self):
        return pde.TrialFunctions(self.function_space)

    @property
    def test_functions(self):
        return pde.TestFunctions(self.function_space)

    def make_functional(self):
        """
        Create linear operator for the total-pressure formulation of Biot's equation of poroelasticity

        div(2*mu*epsilon(u)) + grad(p_T) = f
        p_T = lambda *div(u) - alpha * p_F
        (alpha/lambda)(2*alpha * dp_F/dt + dp_T/dt) - div(kappa*grad(p_T)) = g

        Homogeneous dirichlet conditions are applied to all boundaries, and any additional dirichlet
        conditions should be added before assembly.
        """
        mu = self.shear_modulus
        lambda_ = self.lame_parameter
        alpha = self.biot_coefficient
        kappa = self.hydraulic_conductivity
        f = self.force_field
        g = self.source_field
        dt = self.timestep

        u, pT, pF = self.trial_functions
        v, qT, qF = self.test_functions
        u_n, pT_n, pF_n = pde.split(self.previous_solution)

        conservation_of_momentum = (
            + inner(epsilon(u), epsilon(v))
            - inner(pT, div(v))
            - inner(f, v)
        ) * dx

        total_pressure = (
            - (1/lambda_)*pT * qT
            - div(u) * qT
            + (alpha/lambda_)*pF*qT
        ) * dx

        dpF = pF - pF_n
        dpT = pT - pT_n
        temporal_evolution = (
            + (alpha / lambda_) * (dpT/dt - 2 * alpha * dpF/dt) * qF
            - inner(kappa * grad(pF), grad(qF))
            - g * qF
        ) * dx
        return conservation_of_momentum + total_pressure + temporal_evolution

    def make_preconditioner(self):
        lambda_ = self.lame_parameter
        alpha = self.biot_coefficient
        kappa = self.hydraulic_conductivity
        dt = self.timestep

        u, pT, pF = self.trial_functions
        v, qT, qF = self.test_functions

        preconditioner = (
            + inner(epsilon(u), epsilon(v))
            + pT*qT
            + (1/dt) * (alpha*alpha/lambda_)*pF*qF + inner(kappa*grad(pF), grad(qF))
        )*dx
        return preconditioner
