#!/usr/bin/python3

from mesh import Mesh
mesh = Mesh("z") # a quad mesh with regular grid connectivity
n = mesh.size
u,v = mesh.x[:n*n], mesh.x[n*n:] # the grid is made of n*n verts

for _ in range(128): # Gauss-Seidel iterations solving for zero Laplacian
    for j in range(1, n-1):     # the boundary is fixed, so we iterate
        for i in range(1, n-1): # through interior vertices only
            idx = i+j*n
            u[idx] = (u[idx-1] + u[idx+1] + u[idx-n] + u[idx+n])/4.
            v[idx] = (v[idx-1] + v[idx+1] + v[idx-n] + v[idx+n])/4.
print(mesh)
mesh.show()

