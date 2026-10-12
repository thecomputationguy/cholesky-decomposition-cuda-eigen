#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <Eigen/Dense>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <omp.h>

std::vector<int> read_file(); // forward declaraion of the method to read resolutions
Eigen::MatrixXf make_tridiagonal(int size); // SPD tridiagonal matrix tridiag(-1, 2, -1)
double condition_number_analytic(const std::string& matrix_type, int size); // closed-form 2-norm condition number

int main(int argc, char* argv[])
{
    // Matrix to be solved : "tridiagonal" (default) or "identity"
    std::string matrix_type = (argc > 1) ? argv[1] : "tridiagonal";
    if(matrix_type != "tridiagonal" && matrix_type != "identity")
    {
        std::cerr<<"Unknown matrix type '"<<matrix_type<<"'. Use 'tridiagonal' or 'identity'."<<std::endl;
        return 1;
    }

    std::cout<<"\n***** Starting Cholesky Solvers (matrix : "<<matrix_type<<") *****"<<std::endl;

    std::vector<int> resolutions = read_file();
    Eigen::setNbThreads(omp_get_max_threads()); // To enable multithreaded computation, if and when available.
    std::chrono::steady_clock::time_point start; // start timer
    std::chrono::steady_clock::time_point stop; // stop timer

    std::ofstream out("measurements.csv"); // to store the measurement data
    out<<"Resolution,CPU,GPU,GPU-Speedup,Cond,CPU-RelErr,GPU-RelErr\n";

    // Warm-up : the first cuSOLVER call pays one-time costs (context creation, module loading).
    // Do a small untimed solve so that cost is not attributed to the first measured resolution.
    {
        const int n = 64;
        int lwork, *dinfo;
        cusolverDnHandle_t whandle;
        cusolverDnCreate(&whandle);
        Eigen::MatrixXf I = Eigen::MatrixXf::Identity(n, n);
        thrust::device_vector<float> wA(I.data(), I.data() + I.size());
        thrust::device_vector<float> wb(n, 1.0f);
        cudaMalloc((void **) &dinfo, sizeof(int));
        cusolverDnSpotrf_bufferSize(whandle, CUBLAS_FILL_MODE_LOWER, n, wA.data().get(), n, &lwork);
        thrust::device_vector<float> wWork(lwork);
        cusolverDnSpotrf(whandle, CUBLAS_FILL_MODE_LOWER, n, wA.data().get(), n, wWork.data().get(), lwork, dinfo);
        cusolverDnSpotrs(whandle, CUBLAS_FILL_MODE_LOWER, n, 1, wA.data().get(), n, wb.data().get(), n, dinfo);
        cudaDeviceSynchronize();
        cudaFree(dinfo);
        cusolverDnDestroy(whandle);
    }

    for(int j = 0; j < resolutions.size(); j++)
    {
        int size = resolutions[j];

        // Create CUDA instances and handles
        cudaError cudaStatus;
        cusolverStatus_t cusolverStatus ;
        cusolverDnHandle_t handle;
        cusolverStatus = cusolverDnCreate(&handle);

        /* 
        Create data structures in Eigen. 
        Equation considered is Ax=b, with x_true random and b = A * x_true, so the exact solution is known.
        identity    : A = I, thus x = b.
        tridiagonal : A = tridiag(-1, 2, -1), the SPD 1D Laplacian.
        */
        
        Eigen::MatrixXf A(size, size);
        Eigen::VectorXf b(size), x(size);
        Eigen::VectorXf x_true = Eigen::VectorXf::Random(size);
        A = (matrix_type == "identity") ? Eigen::MatrixXf(Eigen::MatrixXf::Identity(size, size)) : make_tridiagonal(size);
        b = A * x_true;
        x = x_true;

        int Lwork, *d_info;

        // Create device data in Thrust directly from the Eigen host data (no intermediate host copies)
        thrust::device_vector<float> d_A(A.data(), A.data() + A.size());
        thrust::device_vector<float> d_b(b.data(), b.data() + b.size());
        cudaStatus = cudaMalloc((void **) &d_info, sizeof(int));  

        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER ;
        cusolverStatus = cusolverDnSpotrf_bufferSize(handle, uplo, size, d_A.data().get(), size, &Lwork);
        
        thrust::device_vector<float> d_Work(Lwork);

        // Solve on the GPU
        cudaStatus = cudaDeviceSynchronize(); // make sure no earlier work is still in flight
        start = std::chrono::steady_clock::now();

        cusolverStatus = cusolverDnSpotrf(handle, uplo, size, d_A.data().get(), size, d_Work.data().get(), Lwork, d_info);
        cusolverStatus = cusolverDnSpotrs(handle, uplo, size, 1, d_A.data().get(), size, d_b.data().get(), size, d_info);
        cudaStatus = cudaDeviceSynchronize();
        
        stop = std::chrono::steady_clock::now();
        double duration_gpu = std::chrono::duration<double, std::milli>(stop - start).count(); // milliseconds

        // Solve on the CPU
        start = std::chrono::steady_clock::now();

        Eigen::LDLT<Eigen::MatrixXf> ldlt(size);
        ldlt.compute(A);
        x = ldlt.solve(b);

        stop = std::chrono::steady_clock::now();
        double duration_cpu = std::chrono::duration<double, std::milli>(stop - start).count(); // milliseconds
        double speedup = duration_cpu / duration_gpu;

        // Check results : copy the GPU solution (device -> host) directly into an Eigen vector
        Eigen::VectorXf x_gpu(size);
        thrust::copy(d_b.begin(), d_b.end(), x_gpu.data());

        // Error against the known exact solution, and condition number of A
        double rel_err_cpu = (x - x_true).norm() / x_true.norm();
        double rel_err_gpu = (x_gpu - x_true).norm() / x_true.norm();
        double cond = condition_number_analytic(matrix_type, size);

        std::cout<<"\nResolution : "<<size<<std::endl;
        std::cout<<std::fixed<<std::setprecision(3);
        std::cout<<"\tGPU (milliseconds) : "<<duration_gpu;
        std::cout<<"\n\tCPU (milliseconds) : "<<duration_cpu;
        std::cout<<std::defaultfloat<<std::setprecision(6);
        std::cout<<"\n\tDifference (in 2 Norm) : "<<(x - x_gpu).squaredNorm()<<std::endl;
        std::cout<<"\n\tGPU Speedup : "<<std::fixed<<std::setprecision(2)<<speedup<<std::defaultfloat<<std::setprecision(6)<<std::endl;
        std::cout<<"\n\tCondition number (analytic) : "<<cond;
        if(size <= 1000) // O(n^3) eigendecomposition is only affordable for small sizes
        {
            Eigen::VectorXd ev = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(A.cast<double>(), Eigen::EigenvaluesOnly).eigenvalues();
            std::cout<<"\n\tCondition number (numeric)  : "<<ev(size - 1) / ev(0);
        }
        std::cout<<"\n\tRelative error vs exact (CPU) : "<<rel_err_cpu;
        std::cout<<"\n\tRelative error vs exact (GPU) : "<<rel_err_gpu<<std::endl;

        // Write measurements to file
        out<<size<<","<<duration_cpu<<","<<duration_gpu<<","<<speedup<<","<<cond<<","<<rel_err_cpu<<","<<rel_err_gpu<<"\n";

        // Release CUDA resources created in this iteration
        cudaStatus = cudaFree(d_info);
        cusolverStatus = cusolverDnDestroy(handle);
    }

    out.close();    
    return 0;
}


std::vector<int> read_file()
{
    /*
        Method to read resolutions.
    */

    std::string filename = "resolutions.txt";
    std::vector<int> resolutions;
    std::string size;
    std::fstream file(filename, std::ios::in);

    if(file.is_open())
    {
        while(std::getline(file, size))
        {
            resolutions.push_back(stoi(size));
        }
    }

    return resolutions;
}

Eigen::MatrixXf make_tridiagonal(int size)
{
    /*
        Symmetric positive definite tridiagonal matrix tridiag(-1, 2, -1),
        i.e. the 1D Poisson / Laplacian stencil.
    */

    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(size, size);
    for(int i = 0; i < size; i++)
    {
        A(i, i) = 2.0f;
        if(i > 0)        A(i, i - 1) = -1.0f;
        if(i < size - 1) A(i, i + 1) = -1.0f;
    }
    return A;
}

double condition_number_analytic(const std::string& matrix_type, int size)
{
    /*
        Closed-form 2-norm condition number.
        tridiag(-1, 2, -1) has eigenvalues l_k = 2 - 2cos(k*pi/(n+1)), k = 1..n,
        so cond = l_n / l_1 = cot^2(pi / (2(n+1))) ~ 4(n+1)^2 / pi^2.
    */

    if(matrix_type == "identity")
        return 1.0;

    const double pi = 3.14159265358979323846;
    double l_min = 2.0 - 2.0 * std::cos(pi / (size + 1));
    double l_max = 2.0 - 2.0 * std::cos(size * pi / (size + 1));
    return l_max / l_min;
}
