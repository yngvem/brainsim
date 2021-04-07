import sys
from argparse import ArgumentParser
from time import time

import mshr
import numpy as np
from brainsim.biot_system import BiotSystem
import fenics as pde
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

parser = ArgumentParser()
parser.add_argument("mesh")
args = parser.parse_args()

def petsc_to_scipy(petsc_matrix):
    return sparse.csr_matrix(petsc_matrix.getValuesCSR()[::-1])

def dolfin_to_scipy(dolfin_matrix):
    return petsc_to_scipy(pde.as_backend_type(dolfin_matrix).mat())

# Generate mesh
geometry = mshr.Rectangle(pde.Point(-1, -1), pde.Point(1, 1))
mesh = pde.Mesh(args.mesh)

# Physical parameters
mu = pde.Constant(1)
lambda_ = pde.Constant(1)
alpha = pde.Constant(1)
kappa = pde.Constant(1)
f = pde.Constant(np.zeros(mesh.coordinates().shape[1]))
g = pde.Constant(0)
dt = pde.Constant(1)

# Construct dynamic system
system = BiotSystem(
    mesh=mesh,
    shear_modulus=mu,
    lame_parameter=lambda_,
    biot_coefficient=alpha,
    hydraulic_conductivity=kappa,
    force_field=f,
    source_field=g,
    timestep=dt,
)

F = system.make_functional()
preconditioner = system.make_preconditioner()
bcs = []

# Construct functionals
a = pde.lhs(F)
L = pde.rhs(F)
soln = system.solution
prev_soln = system.previous_solution

A, b = pde.assemble_system(a, L, bcs=bcs)
B = pde.assemble_system(preconditioner, L, bcs=bcs)[0]
solver = pde.KrylovSolver("bicgstab", "amg")
solver.set_operators(A, B)

soln.vector()[:] = np.random.standard_normal(len(soln.vector()))
t = 0

print(f"Solving system of size {len(soln.vector())}...")
timer_start = time()
b = pde.assemble(L)
for bc in bcs:
    bc.apply(b)
solver.solve(soln.vector(), b)
prev_soln.assign(soln)

timer_end = time()
print(f"Elapsed {timer_end - timer_start:.2f} seconds")
