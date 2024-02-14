import os
os.environ["OMP_NUM_THREADS"] = "1"
# import numpy as np
from firedrake import *
import math
import time

N = 64
mesh = PeriodicRectangleMesh(7*N, N, 7, 1, direction="x") # L = 3891 km; R = 6370 km

V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V0 = FunctionSpace(mesh, "DG", 0) 

x, y = SpatialCoordinate(mesh)

# define dimensionless parameters
Ro = 10**(-1) ; Re = 10**3 ; B = 10**0 ; C = 10**(-2) ; Pe = 10**3

# define initial condtions
y0 = 1/14 ; y1 = 13/14
alpha = 1.63 # 1/delta phi where phi is 35 degrees
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
c0 = 0.01 ; c1 = 4 ;  c2 = 81 ; x_0 = 3.5; y_2 = 0.5

h0_i = interpolate(h0_c + h0 , V2)
h0_p = interpolate(c0*cos(math.pi*y/2)*exp(-c1*(x  - x_0)**2)*exp(-c2*(y - y_2)**2), V2)
h0_f = interpolate(h0_c + h0 + h0_p, V2) # perturbed initial height

outfile = File("./results/test.pvd")
h0.rename("avg_h")
h0_i.rename("init_h")
h0_p.rename("pert_h")
h0_f.rename("final_h")
u0.rename("init_vel")

outfile.write(h0, h0_i, h0_p, h0_f,u0)
