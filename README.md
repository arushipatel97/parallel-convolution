# Parallel-Convolution
Analyzing & Parallelizing Image Convolution with Cuda, OpenMP, and MPI

This program involves image (2D kernel) on images, specifically of type .png.
By writing and analyzing the convolution code using each of the methods, we are aiming 
to maximize the speedup from the serial convolution.

# Serial
```
make
./conv filename.png width height rgb/gray
```
# Cuda
```
cd cuda
make
./cudaConv filename.png width height rgb/gray
```
# OpenMP

in progress

# MPI

in progress