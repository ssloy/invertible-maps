#!/usr/bin/python3

from mesh import Mesh
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

mesh = Mesh("z") # a quad mesh with regular grid connectivity
n = mesh.nverts

Q = [ np.matrix('-1,-1;1,0;0,0;0,1'), np.matrix('-1,0;1,-1;0,1;0,0'),  # quadratures for
      np.matrix('0,0;0,-1;1,1;-1,0'), np.matrix('0,-1;0,0;1,0;-1,1') ] # every quad corner

def jacobian(U, qc, quad): # evaluate the Jacobian matrix at the given quadrature point
    return np.matrix([[U[quad[0]  ], U[quad[1]  ], U[quad[2]  ], U[quad[3]  ]],
                      [U[quad[0]+n], U[quad[1]+n], U[quad[2]+n], U[quad[3]+n]]]) * Q[qc]

for iter in range(10): # outer L-BFGS loop
    mindet = min( [ np.linalg.det( jacobian(mesh.x, qc, quad) ) for quad in mesh.quads for qc in range(4) ] )
    eps = np.sqrt(1e-6**2 + .04*min(mindet, 0)**2) # the regularization parameter e
    def energy(U): # compute the energy and its gradient for the map u
        F,G = 0, np.zeros(2*n)
        for quad in mesh.quads: # sum over all quads
            for qc in range(4): # evaluate the Jacobian matrix for every quad corner
                J = jacobian(U, qc, quad)
                det = np.linalg.det(J)
                chi  = det/2 + np.sqrt(eps**2 + det**2)/2    # the penalty function
                chip = .5 + det/(2*np.sqrt(eps**2 + det**2)) # its derivative
                f = np.trace(np.transpose(J)*J)/chi # quad corner shape quality
                F += f
                dfdj = (2*J - np.matrix([[J[1,1],-J[1,0]],[-J[0,1],J[0,0]]])*f*chip)/chi
                dfdu = Q[qc] * np.transpose(dfdj) # chain rule for the actual variables
                for i,v in enumerate(quad):
                    if (mesh.boundary[v]): continue # the boundary verts are locked
                    G[v  ] += dfdu[i,0]
                    G[v+n] += dfdu[i,1]
        return F,G
    mesh.x = fmin_l_bfgs_b(energy, mesh.x, factr=1e12)[0] # inner L-BFGS loop
print(mesh)
mesh.show()

