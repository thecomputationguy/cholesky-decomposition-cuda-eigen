#cholesky-decomposition-cuda-eigen
GPU based implementation of a Cholesky Decomposition based linear solver using CUDA Thrust and cuSOLVER, also featuring Eigen for the purpose of verification and runtime comparison. The aim of this repository is to use high-level, possibly template-based APIs to reduce development time and avoid writing boilerplate code for memory management, cleanup etc.

The system being solved is Ax=b. 'A' is set to an identity matrix of appropriate size and thus, the eventual solution becomes 'x = inverse(A) * b = b'. Thus, if everything works as they should, the solution should be identical to the initial vector 'b' (set randomly).

Simply run the 'run_solvers.sh' script and everything is done automatically. However, one needs to assign appropriate permissions to the shell script first. To do that, in a linux terminal, run 'chmod 777 run_solver.sh'. Then run './run_solvers.sh'. Once the runs are finished, a comparison graph is generated for the runtimes and is saved as 'plot_cholesky.png' and the runtimes are stored as a csv file in 'measurements.csv'.
