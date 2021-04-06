import mshr
from brainsim.biot_system import BiotSystem
import fenics as pde
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

num_iterations = 10
compute_condition_number = False

def petsc_to_scipy(petsc_matrix):
    return sparse.csr_matrix(petsc_matrix.getValuesCSR()[::-1])


def dolfin_to_scipy(dolfin_matrix):
    return petsc_to_scipy(pde.as_backend_type(dolfin_matrix).mat())

# Generate mesh
geometry = mshr.Rectangle(pde.Point(-1, -1), pde.Point(1, 1))
mesh = mshr.generate_mesh(geometry, 64)

force_region = "(-0.5 < x[1]) && (0.5 > x[1]) && near(x[0], -1)"
no_displacement = "near(x[0], 1)"
open_boundary = f"!({no_displacement}) && !({force_region})"

mesh_function = pde.MeshFunction('size_t', mesh, 1)
pde.CompiledSubDomain(force_region).mark(mesh_function, 1)
ds_force_region = pde.ds(subdomain_data=mesh_function)(1)

# Physical parameters
nu = 0.479  # Poisson ratio
E = 16 * 1e3  # Bulk modulus
K_wm = 1.4e-14  # Permeability
mu_w = 7.0e-4  # Viscosity

mu_bar = pde.Constant(E / (2 * (1 + nu)))
mu = pde.Constant(1)
lambda_ = pde.Constant(E * nu / ((1 + nu) * (1 - 2*nu))) / (2*mu_bar)
alpha = pde.Constant(1) / (2*mu_bar)
kappa = pde.Constant(K_wm / mu_w) / (2*mu_bar)
f = pde.Constant((0, 0)) / (2*mu_bar)
g = pde.Constant(0) / (2*mu_bar)
boundary_force = pde.Constant((1, 0)) / (2*mu_bar)
dt = pde.Constant(1e-5)

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
bcs = [pde.DirichletBC(system.function_space.sub(0), (0, 0), no_displacement)]

# Add non-homogeneous Neumann boundary conditions
v = system.test_functions[0]
F -= pde.inner(boundary_force, v) * ds_force_region

# Construct functionals
a = pde.lhs(F)
L = pde.rhs(F)
soln = system.solution
prev_soln = system.previous_solution

# Assemble required matrices
A, b = pde.assemble_system(a, L, bcs=bcs)
B = pde.assemble_system(preconditioner, L, bcs=bcs)[0]

if compute_condition_number:
    csr_A = dolfin_to_scipy(A)
    csr_B = dolfin_to_scipy(B)
    print("Computing maximum eigenvalues", flush=True)
    max_eigs = spla.eigsh(csr_A, M=csr_B, which='LM', return_eigenvectors=False)
    print("Computing minimum eigenvalues", flush=True)
    min_eigs = spla.eigsh(csr_A, M=csr_B, sigma=0, return_eigenvectors=False)
    cond = abs(max_eigs).max() / abs(min_eigs).min()
    print(f"Condition number: {cond:.2g}")


solver = pde.KrylovSolver("bicgstab", "amg")
solver.set_operators(A, B)
solver = pde.LUSolver(A)

displacement = pde.XDMFFile(mesh.mpi_comm(), 'displacement.xdmf')
total_pressure = pde.XDMFFile(mesh.mpi_comm(), 'total_pressure.xdmf')
fluid_pressure = pde.XDMFFile(mesh.mpi_comm(), 'fluid_pressure.xdmf')
t = 0
for i in range(num_iterations):
    b = pde.assemble(L)
    for bc in bcs:
        bc.apply(b)
    solver.solve(soln.vector(), b)
    
    diff = prev_soln - soln
    plt.subplot(131)
    plt.gca().clear()
    plt.title(i)
    pde.plot(soln.split()[0])
    plt.subplot(132)
    plt.gca().clear()
    pde.plot(soln.split()[1])
    plt.subplot(133)
    plt.gca().clear()
    pde.plot(soln.split()[2] - prev_soln.split()[2])
    plt.show()

    prev_soln.assign(soln)
    displacement.write(prev_soln.split()[0], t)
    total_pressure.write(prev_soln.split()[1], t)
    fluid_pressure.write(prev_soln.split()[2], t)

    t += dt(0)

displacement.close()
total_pressure.close()
fluid_pressure.close()
