#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <fstream>
#include <chrono>
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

        // Obtain pointers of the Eigen data so that they can be copied into Thrust vectors
        float *A_eigen = A.data(); 
        float *b_eigen = b.data();
        int Lwork, *d_info;

        // Create host and device data in Thrust
        thrust::host_vector<float> mat_A(A_eigen, A_eigen + A.size());
        thrust::host_vector<float> vec_b(b_eigen, b_eigen + b.size());
        thrust::host_vector<float> info;
        thrust::device_vector<float> d_A = mat_A;
        thrust::device_vector<float> d_b = vec_b;    
        cudaStatus = cudaMalloc((void **) &d_info, sizeof(int));  

        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER ;
        cusolverStatus = cusolverDnSpotrf_bufferSize(handle, uplo, size, d_A.data().get(), size, &Lwork);
        
        thrust::host_vector<float> Work(Lwork);
        thrust::device_vector<float> d_Work = Work;

        // Solve on the GPU
        start = std::chrono::steady_clock::now();

        cusolverStatus = cusolverDnSpotrf(handle, uplo, size, d_A.data().get(), size, d_Work.data().get(), Lwork, d_info);
        cusolverStatus = cusolverDnSpotrs(handle, uplo, size, 1, d_A.data().get(), size, d_b.data().get(), size, d_info);
        cudaStatus = cudaDeviceSynchronize();
        
        stop = std::chrono::steady_clock::now();
        auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

        // Solve on the CPU
        start = std::chrono::steady_clock::now();

        Eigen::LDLT<Eigen::MatrixXf> ldlt(size);
        ldlt.compute(A);
        x = ldlt.solve(b);

        stop = std::chrono::steady_clock::now();
        auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        auto speedup = duration_cpu / duration_gpu;

        // Check results
        vec_b = d_b; // get the device vector pointers
        Eigen::Map<Eigen::VectorXf> x_gpu(vec_b.data(), x.size()); // map the device vector to a host container  

        // Error against the known exact solution, and condition number of A
        double rel_err_cpu = (x - x_true).norm() / x_true.norm();
        double rel_err_gpu = (x_gpu - x_true).norm() / x_true.norm();
        double cond = condition_number_analytic(matrix_type, size);

        std::cout<<"\nResolution : "<<size<<std::endl;
        std::cout<<"\tGPU (milliseconds) : "<<duration_gpu / 1000;
        std::cout<<"\n\tCPU (milliseconds) : "<<duration_cpu / 1000;
        std::cout<<"\n\tDifference (in 2 Norm) : "<<(x - x_gpu).squaredNorm()<<std::endl;
        std::cout<<"\n\tGPU Speedup : "<<speedup<<std::endl;
        std::cout<<"\n\tCondition number (analytic) : "<<cond;
        if(size <= 1000) // O(n^3) eigendecomposition is only affordable for small sizes
        {
            Eigen::VectorXd ev = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(A.cast<double>(), Eigen::EigenvaluesOnly).eigenvalues();
            std::cout<<"\n\tCondition number (numeric)  : "<<ev(size - 1) / ev(0);
        }
        std::cout<<"\n\tRelative error vs exact (CPU) : "<<rel_err_cpu;
        std::cout<<"\n\tRelative error vs exact (GPU) : "<<rel_err_gpu<<std::endl;

        // Write measurements to file
        out<<size<<","<<duration_cpu / 1000<<","<<duration_gpu / 1000<<","<<speedup<<","<<cond<<","<<rel_err_cpu<<","<<rel_err_gpu<<"\n";

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
