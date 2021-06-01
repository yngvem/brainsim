import sys
from tqdm import trange
from pathlib import Path
import numpy as np
import fenics as pde
import matplotlib.pyplot as plt
from ufl import nabla_div, dx, inner, nabla_grad

def fenics_quiver(F, mesh_size=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if mesh_size is None:
        mesh_size = F.function_space().mesh().coordinates().max(axis=0)

    x = np.linspace(0, mesh_size[0], 20)
    y = np.linspace(0, mesh_size[1], 20)
    xx, yy = np.meshgrid(x, y)
    uu = np.empty_like(xx).ravel()
    vv = np.empty_like(xx).ravel()
    for i, (x, y) in enumerate(zip(xx.ravel(), yy.ravel())):
        uu[i] = F(x, y)[0]
        vv[i] = F(x, y)[1]

    ax.quiver(
        xx,
        yy,
        uu.reshape(xx.shape),
        vv.reshape(xx.shape),
        np.sqrt(uu**2 + vv**2).reshape(xx.shape),
        cmap="magma",
    )
    ax.set_title(f"U: {uu.max():.1g}, {uu.min():.1g}, V: {vv.max():.1g}, {vv.min():.1g}")
    ax.axis("equal")


def fenics_contour(F, mesh_size=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if mesh_size is None:
        mesh_size = F.function_space().mesh().coordinates().max(axis=0)

    x = np.linspace(0, mesh_size[0], 100)
    y = np.linspace(0, mesh_size[1], 100)
    xx, yy = np.meshgrid(x, y)
    zz = np.empty_like(xx).ravel()
    for i, (x, y) in enumerate(zip(xx.ravel(), yy.ravel())):
        zz[i] = F((x, y))

    vmax = np.max(np.abs(zz))
    vmin = -vmax
    ax.contourf(xx, yy, zz.reshape(xx.shape), vmin=-1, vmax=1, cmap="coolwarm", levels=100)
    ax.set_title(f"{zz.max():1f}, {zz.min():.1f}")
    ax.axis("equal")

#%% 
nT = int(sys.argv[1])
nX = int(sys.argv[2])
mesh = pde.UnitSquareMesh(nX, nX)
mesh.coordinates()[:] *= 2*np.pi
P2 = pde.VectorElement("CG", mesh.ufl_cell(), 2)
P1 = pde.FiniteElement("CG", mesh.ufl_cell(), 1)
P2P1P1 = pde.MixedElement([P2, P1, P1])
W = pde.FunctionSpace(mesh, P2P1P1)


#%% Set boundary conditions
dirichlet = "on_boundary"


#%% Physical parameters
mu = pde.Constant(0.5)
lambda_ = pde.Constant(1)
alpha = pde.Constant(1)
kappa = pde.Constant(1)
dt = pde.Constant(2*np.pi / nT)


#%% Define U and P
t = pde.Constant(np.pi/2)
x = pde.SpatialCoordinate(mesh)
n = pde.FacetNormal(mesh)
zero = pde.Constant(0)
epsilon = pde.nabla_grad

U = pde.as_vector((
    pde.Constant(1)*pde.sin(x[0])*pde.sin(x[1])*pde.cos(t),
    pde.Constant(1)*pde.sin(x[0])*pde.sin(x[1])*pde.cos(t),
))
P = pde.Constant(1)*pde.sin(x[0])*pde.sin(x[1])*pde.cos(t)
U_dot = pde.as_vector((
    -pde.Constant(1)*pde.sin(x[0])*pde.sin(x[1])*pde.sin(t),
    -pde.Constant(1)*pde.sin(x[0])*pde.sin(x[1])*pde.sin(t),
))
P_dot = -pde.Constant(1)*pde.sin(x[0])*pde.sin(x[1])*pde.sin(t)

P_total = P - nabla_div(U)
P_total_dot = P_dot - nabla_div(U_dot)


#%% Setup manufactured solution
rhs_V = pde.VectorFunctionSpace(mesh, 'CG', 4)
rhs_Q = pde.FunctionSpace(mesh, 'CG', 4)
f = -nabla_div(nabla_grad(U)) + nabla_grad(P_total)

g = P_total_dot - 2*P_dot + nabla_div(nabla_grad(P))

true_u = pde.project(U, rhs_V)
true_p_total = pde.project(P_total, rhs_Q)
true_p = pde.project(P, rhs_Q)


#%% Setup bilinear form
u, pT, pF = pde.TrialFunctions(W)
v, qT, qF = pde.TestFunctions(W)

solution = pde.Function(W)
previous_solution = pde.Function(W)
#u_n, pT_n, pF_n = previous_solution.split()

# dpT = pT - pT_n
# dpF = pF - pT_n
F = (
      inner(nabla_grad(u), nabla_grad(v))*dx - pT*nabla_div(v)*dx                                                           - inner(f, v)*dx
    - nabla_div(u)*qT*dx                     - pT*qT*dx           + pF*qT*dx
                                             + pT*qF*dx           - 2*pF*qF*dx - dt*inner(nabla_grad(pF), nabla_grad(qF))*dx - dt*g*qF*dx
)
a, L = pde.lhs(F), pde.rhs(F)
def timestep_rhs(previous_solution):
    return qF * (previous_solution.split()[1] - 2*previous_solution.split()[2]) * dx

bcs = [
    pde.DirichletBC(W.sub(0), true_u, "on_boundary"),
    pde.DirichletBC(W.sub(2), true_p, "on_boundary"),
]


#%% Setup initial conditions
a_initial = inner(u, v)*dx + inner(pT, qT)*dx + inner(pF, qF)*dx
L_initial = inner(true_u, v)*dx + inner(true_p_total, qT)*dx + inner(true_p, qF)*dx
pde.solve(a_initial == L_initial, solution)


#%% Solve for some iterations
A, b = pde.assemble_system(a, L, bcs=bcs)
solver = pde.LUSolver()
solver.set_operator(A)
for i in trange(nT):
    t.assign(t(0) + dt(0))

    true_u = pde.project(U, rhs_V)
    true_p_total = pde.project(P_total, rhs_Q)
    true_p = pde.project(P, rhs_Q)

    b = pde.assemble(L + timestep_rhs(previous_solution))
    for bc in bcs:
        bc.apply(b)

    solver.solve(solution.vector(), b)


    plt.subplot(321)
    fenics_quiver(true_u)
    plt.subplot(322)
    fenics_quiver(solution.split()[0])
    plt.subplot(323)
    fenics_contour(true_p_total)
    plt.subplot(324)
    fenics_contour(solution.split()[1])
    plt.subplot(325)
    fenics_contour(true_p)
    plt.subplot(326)
    fenics_contour(solution.split()[2])
    plt.show()
sys.exit(0)

if not Path("results.csv").is_file():
    with open("results.csv", "w") as f:
        f.write("nT,nX,u,pT,pF\n")

with open("results.csv", "r") as f:
    data = f.read()

err_u = pde.errornorm(true_u, solution.split()[0], 'h1')
err_pT = pde.errornorm(true_p_total, solution.split()[1], 'l2')
err_pF =  pde.errornorm(true_p, solution.split()[2], 'l2')

with open("results.csv", "w") as f:
    f.write(f"{data}" + f"{nT},{nX},{err_u},{err_pT},{err_pF}\n")

