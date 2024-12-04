# Parallel Computing (CP)

This repository contains files related to the Parallel Computing (CP) course in the Computer Science Bachelor's program at PUC Minas. The activities include assignments provided by Henrique Cota de Freitas, professor of CP, as well as self-assigned tasks.

---

## Projects

Parallelize a sequential version of parallel artificial intelligence application in C or C++.  
The application must fit one of three artificial intelligence categories:
- decision tree.
- grouping.
- neural network.  

### First Project - Parallel (OpenMP && MPI)

Parallelize the application using hybrid MPI/OpenMP.  

`Requirements`:
- Group must use a real input dataset that demands at least 10 seconds of execution time on the parcode server on sequential version.  
(Parcode server is a dedicated server used in-class for this course, to standardize the tests).
- Developing the sequential version is optional, an available open-source code may be used, provided a link to the source code is included.  
- Create a readme.txt file with compilation and execution instructions as well as explanation of the application.  
- For OpenMP-only version, include comments with execution times, for sequential and parallel versions (1, 2, 4 and 8 threads)  
- For MPI version, state execution times as above for the following configurations, openMP is not allowed for this.  
  
| Processes | Threads | Time |
| --- | --- | --- |
| 1 | 4 | time |
| 2 | 2 | time |
| 4 | 0 | time |

### Second Project - GPU (OpenMP && CUDA)

Parallelize the code developed in Project 01 to run on a GPU using both OpenMP and CUDA versions.

`Requirements`:
- The group must parallelize the existing solution for the GPU, creating two separate versions: one using OpenMP and the other using CUDA.  
- The group must perform performance and scalability tests and compare the results between the GPU versions and the original sequential version from Project 01.  

---

## Tasks

All tasks are compelted in C using linux GCC compiler, and each .c file includes code execution times.

### Task2

Given a code, use OpenMP MAP to parallelize it by adding directives only, without altering the code.

### Task3

Implement a parallel code in OpenMP using REDUCE pattern and process scheduling to count prime numbers between 2 and n, where n is an integer <= 1.000.000.000. The provided code uses Sieve of Eratosthenes.

The speed-up must be higher than 1.3, and the custom scheduling policy should show better performance than the default policy.

### Task4

Given an MPI code, complete it with `send` and `receive` operations to make it function correctly

Collective communication operations are not allowed.

### Task6

Complete a given MPI code to accept any number of processes using collective communication.

### Task7

Given a code for matrix multiplication, parallelize it and evaluate performance.

The code should be tested sequentially, in multicore parallel mode, and in GPU parallel mode.
- GPU parallelization use `distribute`, `distribute parallel for`, and `distribute parallel for simd`.

All versions must be compiled with the `-O3` optimization flag and GCC 8. The multicore solution should show performance improvement (speedup) over the sequential.

### Task8

Using the same matrix multiplication for Task7, parallelize for GPU using OpenMP and CUDA.

Versions to implement:
- Sequential
- Multicore parallel
- Gpu parallel with OpenMP
- Gpu parallel with CUDA

For the GPU version `warps_lauched` and `warp_execution_efficiency` should be commented on when tested, as outputted by `nvprof`. All test run times must be stated, with compilation using `-O3` and GCC 8. Use `nvcc` for CUDA.

Provided Hints:

I - Matrix multiplication (mm) uses two dimensions (x/y | i/j), representing rows and columns, so the kernel should have two indices:
```C
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```
II - Since mm involves multiple nested loops, granularity is by output element, meaning only the inner `"k"` loop will run in the kernel.

III - `dimGrid` and `dimBlock` should use the second dimension (repeat the first in the second aargument).

IV - Ensure threads are not accessing non-existant elements based on both row and column indices within the kernel.
