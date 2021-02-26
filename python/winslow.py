#!/usr/bin/python3

from mesh import Mesh

mesh = Mesh("z") # a quad mesh with regular grid connectivity
n = mesh.size

u,v = mesh.x[:n*n], mesh.x[n*n:] # the grid is made of n*n verts
def g11(i,j): # metric tensor estimation via finite differences
    return (u[i+1+j*n]-u[i-1+j*n])**2/4. + (v[i+1+j*n]-v[i-1+j*n])**2/4.
def g22(i,j):
    return (u[i+j*n+n]-u[i+j*n-n])**2/4. + (v[i+j*n+n]-v[i+j*n-n])**2/4.
def g12(i,j):
    return (u[i+1+j*n]-u[i-1+j*n])*(u[i+j*n+n]-u[i+j*n-n])/4. + \
           (v[i+1+j*n]-v[i-1+j*n])*(v[i+j*n+n]-v[i+j*n-n])/4.
for _ in range(128): # Gauss-Seidel iterations, zero Laplacian of the inverse map
     for j in range(1, n-1):     # the boundary is fixed, so we iterate
         for i in range(1, n-1): # through interior vertices only
             a,b,c = g22(i,j), 2*g22(i,j)+2*g11(i,j), g22(i,j)
             d = g11(i,j)*(u[i+j*n+n] + u[i+j*n-n]) - 2*g12(i,j)* \
                 (u[i+1+j*n+n] + u[i-1+j*n-n] - u[i-1+j*n+n] - u[i+1+j*n-n])/4.
             e = g11(i,j)*(v[i+j*n+n] + v[i+j*n-n]) - 2*g12(i,j)* \
                 (v[i+1+j*n+n] + v[i-1+j*n-n] - v[i-1+j*n+n] - v[i+1+j*n-n])/4.
             u[i+j*n] = (d + a*u[i-1+j*n] + c*u[i+1+j*n])/b # actual Gauss-Seidel
             v[i+j*n] = (e + a*v[i-1+j*n] + c*v[i+1+j*n])/b # linear system update
print(mesh)
mesh.show()

