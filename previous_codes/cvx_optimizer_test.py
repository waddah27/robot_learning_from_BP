from cvxopt import matrix, solvers
import xalglib
import numpy as np

def solve_qp_cvxopt():
    # Number of variables
    n = 3

    # Quadratic term: P matrix (symmetric)
    P = matrix([[2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0]])

    # Linear term: q vector
    q = matrix([-1.0, -1.0, -1.0])

    # Inequality constraints (Gx <= h)
    # Assuming x >= 0 for all x, thus G should be -I (negative identity matrix)
    G = matrix([[-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]])
    h = matrix([0.0, 0.0, 0.0])

    # Solve QP problem
    sol = solvers.qp(P, q, G, h)

    # Print solution
    print("Solution:")
    print(sol['x'])


def solve_qp_bleic_example(a, b, c, ct, scale):
    #
    # This example demonstrates minimization of F(x0,x1) = x0^2 + x1^2 -6*x0 - 4*x1
    # subject to linear constraint x0+x1<=2
    #
    # Exact solution is [x0,x1] = [1.5,0.5]
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

    isupper = True

    # create solver, set quadratic/linear terms
    state = xalglib.minqpcreate(2)
    xalglib.minqpsetquadraticterm(state, a, isupper)
    xalglib.minqpsetlinearterm(state, b)
    xalglib.minqpsetlc(state, c, ct)

    # Set scale of the parameters.
    # It is strongly recommended that you set scale of your variables.
    # Knowing their scales is essential for evaluation of stopping criteria
    # and for preconditioning of the algorithm steps.
    # You can find more information on scaling at http://www.xalglib.net/optimization/scaling.php
    #
    # NOTE: for convex problems you may try using minqpsetscaleautodiag()
    #       which automatically determines variable scales.

    xalglib.minqpsetscale(state, scale)

    # Solve problem with QP-BLEIC solver.
    xalglib.minqpsetalgobleic(state, 0.0, 0, 0, 0)
    xalglib.minqpoptimize(state)
    x, rep = xalglib.minqpresults(state)
    print(x) # expected [1.5,0.5]

P = [[2,0],[0,6]]
q = [-1, -1]
A = [[1.0,1.0,1.0]]
bb = [1]


a = [[2,0],[0,2]]
b = [-6,-4]
s = [1,1]
c = [[1.0,1.0,2.0]]
ct = [-1]

if __name__=="__main__":
    # solve_qp_bleic_example(a,b,c,ct,s)
    # solve_qp_bleic_example(a=P,b=q,c=A,ct=bb,scale=s)
    solve_qp_cvxopt()

