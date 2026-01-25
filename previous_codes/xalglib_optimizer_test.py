import xalglib
import numpy as np


#
# This example demonstrates minimization of F(x0,x1) = x0^2 + x1^2 -6*x0 - 4*x1
# subject to Box constraints 0<=x0<=2.5, 0<=x1<=2.5
#
# Exact solution is [x0,x1] = [2.5,2]
#
# IMPORTANT: this solver minimizes  following  function:
#
#     f(x) = 0.5*x'*A*x + b'*x.
#
# Note that quadratic term has 0.5 before it. So if you want to minimize
# quadratic function, you should rewrite it in such way that quadratic term
# is multiplied by 0.5 too.
#
# For example, our function is f(x)=x0^2+x1^2+..., but we rewrite it as
#
#     f(x) = 0.5*(2*x0^2+2*x1^2) + ....
#
# and pass diag(2,2) as quadratic term - NOT diag(1,1)!
#
n = 3
a = [[2,0],[0,2]]
b = [-6,-4]
s = [1,1, 1]
bndl = [200,200, 200]
bndu = [2000,2000, 2000]
isupper = True
x_tilde = np.array([0.3, 0.2, 0.22])
x_t_list = x_tilde.tolist()
x_nt_list = (-1 * x_tilde).tolist()
F_d = np.array([0, 200, 20000])
x_tilde2 = x_tilde@x_tilde.T
k_min = np.eye(n)*200
k_max = np.eye(n)*2000
Q = np.eye(n)*3200
R = np.eye(n)
A = 2 * x_tilde2 * Q + R
C = [[1,0, 60], [-1,0, 60]]
ct = [1,1]

# b = -x_tilde * F_d.T * Q - k_min.T * R
b = [0, 0, 0]
A_list = A.tolist()
# create solver, set quadratic/linear terms
state = xalglib.minqpcreate(n)
xalglib.minqpsetquadraticterm(state, A_list, isupper)
xalglib.minqpsetlinearterm(state, b)
xalglib.minqpsetbc(state, bndl, bndu)
xalglib.minqpsetlc(state, C, ct)

# Set scale of the parameters.
# It is strongly recommended that you set scale of your variables.
# Knowing their scales is essential for evaluation of stopping criteria
# and for preconditioning of the algorithm steps.
# You can find more information on scaling at http://www.alglib.net/optimization/scaling.php
#
# NOTE: for convex problems you may try using minqpsetscaleautodiag()
#       which automatically determines variable scales.
# xalglib.minqpsetscale(state, s)

#
# Solve problem with the sparse interior-point method (sparse IPM) solver.
#
# This solver is intended for large-scale sparse problems with box and linear
# constraints, but it will work on such a toy problem too.
#
# Default stopping criteria are used, Newton phase is active.
#
xalglib.minqpsetalgobleic(state, 0.0, 0.0, 0.0, 0)
xalglib.minqpoptimize(state)
x, rep = xalglib.minqpresults(state)
print(x) # expected [2.5,2]

#
# Solve problem with dense IPM solver.
#
# This solver is optimized for problems with dense linear constraints and/or
# dense quadratic term.
#
# Default stopping criteria are used.
#
xalglib.minqpsetalgodenseipm(state, 0.0)
xalglib.minqpoptimize(state)
x, rep = xalglib.minqpresults(state)
print(x) # expected [2.5,2]