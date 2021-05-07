# Mesh smoothing/untangling

This repository contains the source code for 2d/3d constrained boundary mesh untangling.

This code successfully passes the entire Locally Injective Mappings Benchmark [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3827969.svg)](https://doi.org/10.5281/zenodo.3827969)

For the initial testing purposes, we provide a copy of two example problems (`david-A-input.obj` and `armadillo-598-init.vtk`) taken from the benchmark.
Challenge the code with your data!

# Compile and run:
```sh
git clone --recurse-submodules https://github.com/ssloy/invertible-maps &&
cd invertible-maps/cpp &&
mkdir build &&
cd build &&
cmake .. &&
make -j &&
./untangle2d ../david-A-input.obj result2d.obj &&
./untangle3d ../armadillo-598-init.vtk ../armadillo-598-rest.vtk result3d.vtk
```

