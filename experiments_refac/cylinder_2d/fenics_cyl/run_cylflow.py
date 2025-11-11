from fenics import *
from fenics import near
from mshr import *
import numpy as np
import pathlib
import os
import time
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from datasets.cylinder_2d.load_data import load_data
from experiments_refac.cylinder_2d.utils import CaseConfig, DEFAULT_CONFIG

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
TRAIN_VAL_TEST = "test"

config = DEFAULT_CONFIG
data = load_data(config.data_file)

mu = data[f"{TRAIN_VAL_TEST}_mu"].cpu().detach().numpy()
t = data[f"{TRAIN_VAL_TEST}_t"].cpu().detach().numpy()
x = data[f"{TRAIN_VAL_TEST}_x"].cpu().detach().numpy()
f = data[f"{TRAIN_VAL_TEST}_f"].cpu().detach().numpy()

start_time = time.time()

# Simulation parameters
T = t[0, -1] - t[0, 0] # final time
dt = 0.001          # time step size
dt_samp = t[0, 1] - t[0, 0]

every_n_samp = int(round(dt_samp / dt))
num_steps = int(round(T / dt)) + 1  # number of time steps
num_steps_samp = int(round(T / dt_samp)) + 1

rho = 1.0         # density
nu = 0.00078125 # kinematic viscosity (mu/rho)
mu = nu * rho  # dynamic viscosity

# Domain and mesh
channel_length = 4.0
channel_height = 1.0
cylinder_x = 0.5
cylinder_y = 0.5
cylinder_center = Point(cylinder_x, cylinder_y)
cylinder_radius = 0.0625
N_bulk = 12 # 12 for coarse, 133 for fine

# Create mesh
domain = Rectangle(Point(0, 0), Point(channel_length, channel_height))\
        - Circle(cylinder_center, cylinder_radius, 16)
mesh = generate_mesh(domain, N_bulk)
print("Number of cells in the mesh:", mesh.num_cells())

fig = plt.figure(figsize=(16, 4))
plot(mesh)
plt.title("Generated Mesh")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect('equal')
plt.savefig(os.path.join(CURR_DIR, f'mesh_{mesh.num_cells()}.pdf'), dpi=300)
plt.show()


# Function spaces
V = VectorFunctionSpace(mesh, 'P', 2)  # velocity
Q = FunctionSpace(mesh, 'P', 1)        # pressure

# Boundary conditions
inlet_velocity = Constant((1.0, 0.0))  # inlet velocity
noslip = Constant((0, 0))

# Boundary markers
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], channel_length) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0) or near(x[1], channel_height)) and on_boundary

class Cylinder(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(np.sqrt((x[0] - cylinder_x)**2 + (x[1] - cylinder_y)**2), cylinder_radius, 0.01)

# Apply boundary conditions
bc_inlet = DirichletBC(V, inlet_velocity, Inlet())
bc_walls = DirichletBC(V.sub(1), Constant(0.0), Walls())
bc_cylinder = DirichletBC(V, noslip, Cylinder())
bc_outlet = DirichletBC(Q, Constant(0.0), Outlet())

bcu = [bc_inlet, bc_walls, bc_cylinder]
bcp = [bc_outlet]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions
u_n = Function(V)
u_ = Function(V)
p_n = Function(Q)
p_ = Function(Q)

# === Load initial velocity field ===
# Replace with your file path or array as needed
initial_velocity = np.transpose(x[0, 0], [1, 2, 0])  # shape: (80, 320, 2)
print(initial_velocity.shape)

# Plot velocity x-component
plt.figure(figsize=(10, 3))
plt.imshow(initial_velocity[:, :, 0], extent=(0, channel_length, 0, channel_height), origin='lower', cmap='coolwarm', aspect='equal')
plt.colorbar(label='Velocity x-component')
plt.title('Velocity x-component initial condition')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(CURR_DIR, 'velocity_x_component_init.png'))

# Plot velocity y-component
plt.figure(figsize=(10, 3))
plt.imshow(initial_velocity[:, :, 1], extent=(0, channel_length, 0, channel_height), origin='lower', cmap='coolwarm', aspect='equal')
plt.colorbar(label='Velocity y-component')
plt.title('Velocity y-component initial condition')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(CURR_DIR, 'velocity_y_component_init.png'))

# Grid definition (should match the one used for sampling)
nx, ny = 320, 80
x_vals = np.linspace(0, channel_length, nx)
y_vals = np.linspace(0, channel_height, ny)

# === Define interpolator expression ===
class InterpolatedVelocity(UserExpression):
    def __init__(self, data, x_vals, y_vals, **kwargs):
        super().__init__(**kwargs)
        self.interp_u = RegularGridInterpolator((y_vals, x_vals), data[:, :, 0], bounds_error=False, fill_value=0.0)
        self.interp_v = RegularGridInterpolator((y_vals, x_vals), data[:, :, 1], bounds_error=False, fill_value=0.0)

    def eval(self, value, x):
        pt = [x[1], x[0]]  # [y, x] ordering
        value[0] = self.interp_u(pt)
        value[1] = self.interp_v(pt)

    def value_shape(self):
        return (2,)

# === Interpolate into function space ===
velocity_expr = InterpolatedVelocity(initial_velocity, x_vals, y_vals, degree=2)
u_n.interpolate(velocity_expr)

# Make sure `u_` starts the same way (optional)
u_.assign(u_n)

###
X, Y = np.meshgrid(x_vals, y_vals)

# Evaluate u_n on the uniform grid
u_x = np.zeros_like(X)
u_y = np.zeros_like(Y)

for i in range(ny):
    for j in range(nx):
        pt = [X[i, j], Y[i, j]]
        try:
            u_val = u_n(pt)
        except:
            u_val = (0.0, 0.0)
        u_x[i, j] = u_val[0]
        u_y[i, j] = u_val[1]

# === Plot velocity magnitude ===
velocity_magnitude = np.sqrt(u_x**2 + u_y**2)

plt.figure(figsize=(10, 3))
plt.imshow(velocity_magnitude, extent=(0, channel_length, 0, channel_height),
           origin='lower', cmap='viridis', aspect='equal')
plt.colorbar(label='Velocity Magnitude')
plt.title('Interpolated Initial Velocity Magnitude')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(CURR_DIR, 'interp_init.png'))
###

# Kinematic viscosity from Re = U * D / nu
D = 2 * cylinder_radius
U_mean = 1.0
Re = U_mean * D / nu
print(f"Reynolds number: {Re}")

# Define expressions used in variational forms
U = 0.5*(u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0))

# Tentative velocity step
F1 = rho*dot((u - u_n) / dt, v)*dx + \
     rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx + \
     mu*inner(grad(u), grad(v))*dx - \
     dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure correction
a2 = dot(grad(p), grad(q))*dx
L2 = dot(grad(p_n), grad(q))*dx - (rho/dt)*div(u_)*q*dx

# Velocity correction
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - dt/rho*dot(grad(p_ - p_n), v)*dx

X, Y = np.meshgrid(x_vals, y_vals)
points = np.array([X.flatten(), Y.flatten()]).T

u_resamp = np.zeros((t.shape[1], ny, nx, 2))  # [time, y, x, component]
p_resamp = np.zeros((t.shape[1], ny, nx))     # [time, y, x]

# Time-stepping
t = 0

for i, pt in enumerate(points):
    try:
        u_val = u_(pt)
        p_val = p_(pt)
    except:
        u_val = (0, 0)
        p_val = 0
    ix = i % nx
    iy = i // nx
    u_resamp[0, iy, ix, :] = u_val
    p_resamp[0, iy, ix] = p_val

for n in range(1, num_steps):
    t += dt

    # Suppress printing from solve calls
    set_log_level(30)  # Set log level to WARNING to suppress INFO messages

    # Step 1: Tentative velocity
    solve(a1 == L1, u_, bcu)

    # Step 2: Pressure correction
    solve(a2 == L2, p_, bcp)

    # Step 3: Velocity correction
    solve(a3 == L3, u_, bcu)

    # Restore log level
    set_log_level(20)  # Set log level back to INFO

    # Resample velocity and pressure at specified points
    if n % every_n_samp == 0:
        for i, pt in enumerate(points):
            try:
                u_val = u_(pt)
                p_val = p_(pt)
            except:
                u_val = (0, 0)
                p_val = 0
            ix = i % nx
            iy = i // nx
            u_resamp[(n // every_n_samp), iy, ix, :] = u_val
            p_resamp[(n // every_n_samp), iy, ix] = p_val

    # Update
    u_n.assign(u_)
    p_n.assign(p_)

    if n % 50 == 0:
        print(f"Step {n}, Time {t:.2f}")

total_time = time.time() - start_time
print(f"Total simulation time: {total_time:.2f} seconds")

# Plot velocity x-component
plt.figure(figsize=(10, 3))
plt.imshow(u_resamp[0, :, :, 0], extent=(0, channel_length, 0, channel_height), origin='lower', cmap='coolwarm', aspect='equal')
plt.colorbar(label='Velocity x-component')
plt.title('Velocity x-component at Final Time Step')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(CURR_DIR, 'velocity_x_component_0.png'))

# Plot velocity y-component
plt.figure(figsize=(10, 3))
plt.imshow(u_resamp[0, :, :, 1], extent=(0, channel_length, 0, channel_height), origin='lower', cmap='coolwarm', aspect='equal')
plt.colorbar(label='Velocity y-component')
plt.title('Velocity y-component at Final Time Step')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(CURR_DIR, 'velocity_y_component_0.png'))

# Plot pressure field
plt.figure(figsize=(10, 3))
plt.imshow(p_resamp[0, :, :], extent=(0, channel_length, 0, channel_height), origin='lower', cmap='coolwarm', aspect='equal')
plt.colorbar(label='Pressure')
plt.title('Pressure at Final Time Step')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(CURR_DIR, 'pressure_field_0.png'))

# Plot velocity x-component
plt.figure(figsize=(10, 3))
plt.imshow(u_resamp[-1, :, :, 0], extent=(0, channel_length, 0, channel_height), origin='lower', cmap='coolwarm', aspect='equal')
plt.colorbar(label='Velocity x-component')
plt.title('Velocity x-component at Final Time Step')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(CURR_DIR, 'velocity_x_component_final.png'))

# Plot velocity y-component
plt.figure(figsize=(10, 3))
plt.imshow(u_resamp[-1, :, :, 1], extent=(0, channel_length, 0, channel_height), origin='lower', cmap='coolwarm', aspect='equal')
plt.colorbar(label='Velocity y-component')
plt.title('Velocity y-component at Final Time Step')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(CURR_DIR, 'velocity_y_component_final.png'))

# Plot pressure field
plt.figure(figsize=(10, 3))
plt.imshow(p_resamp[-1, :, :], extent=(0, channel_length, 0, channel_height), origin='lower', cmap='coolwarm', aspect='equal')
plt.colorbar(label='Pressure')
plt.title('Pressure at Final Time Step')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(CURR_DIR, 'pressure_field_final.png'))

print("Simulation completed successfully!")
print(u_resamp.shape, p_resamp.shape)

# Save results
np.savez(os.path.join(CURR_DIR, 'cylinder_flow_results.npz'),
         u_resamp=u_resamp,  # Downsample for storage
         p_resamp=p_resamp
)