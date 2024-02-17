import os
os.environ["OMP_NUM_THREADS"] = "1"
# import numpy as np
from firedrake import *
import math
import time

Ny = 128 # resolution 43 km
Nx = 5*Ny
mesh = PeriodicRectangleMesh(Nx, Ny, 5, 1, direction="x")

V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 0) 

x, y = SpatialCoordinate(mesh)

# define dimensionless parameters
Ro = 0.29 ; Re = 4*10**5 ; B = 2.4 ; C = 0.06 ; Pe = 4*10**5

# define initial condtions
y0 = 4/35 ; y1 = 31/35
alpha = 1.14 # 18/5pi 
u0_1 = conditional(Or(y <= y0, y >= y1), 0.0, exp(alpha**2/((y - y0)*(y - y1)))*exp(4*alpha**2/(y1 - y0)**2))
u0_2 = 0.0

u0 = project(as_vector([u0_1, u0_2]), V1)
g = project(as_vector([u0_2, -(C/Ro)*(1 + B*y)*u0_1]), V1)

f = interpolate(div(g), V0) # i tried project f on V2, doesn't make any difference on the solution

h0 = TrialFunction(V2)
q = TestFunction(V2)

a = -inner(grad(h0), grad(q))*dx
L = f*q*dx

h0 = Function(V2) # geostrophic height
nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD) # this is required with Neumann bcs
solve(a == L, h0, nullspace=nullspace)

# height perturbation
h0_c = 1.0
c0 = 0.01 ; c1 = 9 ;  c2 = 169 ; x_0 = 2.5; y_2 = 0.5

h0_b = interpolate(h0_c + h0 , V2)
h0_p = interpolate(c0*cos(math.pi*y/2)*exp(-c1*(x  - x_0)**2)*exp(-c2*(y - y_2)**2), V2)
h0_f = interpolate(h0_c + h0 + h0_p, V2) # perturbed initial height

outfile = File("./results/test.pvd")
h0.rename("avg_h")
h0_b.rename("balanced_h")
h0_p.rename("perturbation")
h0_f.rename("h_after_pertb")
u0.rename("init_vel")

outfile.write(h0, h0_b, h0_p, h0_f,u0)
