# cholesky-decomposition-cuda-eigen
GPU based implementation of a Cholesky Decomposition based linear solver using CUDA Thrust and cuSOLVER, also featuring Eigen for the purpose of verification and runtime comparison. The aim of this repository is to use high-level, possibly template-based APIs to reduce development time and avoid writing boilerplate code for memory management, cleanup etc.

The system being solved is Ax=b. By default, 'A' is the symmetric positive definite tridiagonal matrix tridiag(-1, 2, -1), and 'b' is computed as 'A * x_true' for a random 'x_true'. Thus, if everything works as it should, both the CPU and GPU solutions should match 'x_true' up to the accuracy allowed by the condition number of 'A'. An identity matrix 'A = I' (where the solution is simply 'x = b') can also be selected, see below.

The problem sizes are defined in the file 'resolutions.txt'. To add another resolution, just append it below the last one.

To be able to run everything properly, one needs the following libraries and ensure that the NVCC compiler is able to find their include paths. This should not be an issue for the CUDA libraries but might be for the external libraries such as Eigen.

    1. CUDA
    2. cuBLAS
    3. cuSOLVER
    4. CUDA Thrust
    5. Eigen
    6. openMP
    6. Pandas and Matplotlib for plotting

Simply run the 'run_solvers.sh' script (Linux) or 'run_solvers.bat' (Windows) and everything is done automatically. On Linux, one needs to assign appropriate permissions to the shell script first. To do that, in a linux terminal, run 'chmod 777 run_solvers.sh'. Then run './run_solvers.sh'. Once the runs are finished, a comparison graph is generated for the runtimes and is saved as 'plot_cholesky.png' and the runtimes are stored as a csv file in 'measurements.csv'.

The include paths for Eigen and the OpenMP thread counts are set inside the scripts, so adjust them to your machine if needed.

## Choosing the matrix

The executable takes an optional argument to choose the matrix being solved:

    cholesky.exe tridiagonal    (default)
    cholesky.exe identity

For both, a random 'x_true' is generated and 'b = A * x_true', so the exact solution is always known.

1. `tridiagonal` : A = tridiag(-1, 2, -1), the symmetric positive definite 1D Laplacian. This is a non-trivial and ill-conditioned problem. Its condition number is cot^2(pi / (2(n+1))), roughly 4(n+1)^2 / pi^2, which reaches about 1e7 at n = 5000 and 1.6e8 at n = 20000. Since the solvers work in single precision, expect the accuracy to degrade at the larger sizes.
2. `identity` : A = I, so x = b. This is a trivial case that is useful as a sanity check.

## Output

For every resolution, the following are printed and written to 'measurements.csv':

    Resolution, CPU, GPU, GPU-Speedup, Cond, CPU-RelErr, GPU-RelErr

CPU and GPU are runtimes in milliseconds (fractional), GPU-Speedup is CPU / GPU, Cond is the 2-norm condition number of A (computed analytically, and also cross-checked numerically on the console for n <= 1000) and the last two columns are the relative errors of the CPU and GPU solutions with respect to the exact solution.

An untimed warm-up solve is performed before the measurements so that the one-time CUDA/cuSOLVER initialisation cost is not attributed to the first resolution. Only the factorization and the solve are timed; data transfers are excluded.
