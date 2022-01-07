#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <Eigen/Dense>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <omp.h>

#define NUM_THREADS 6;

int main(int argc, char* argv[])
{
    std::cout<<"\n***** Starting Cholesky Solvers *****"<<std::endl;
    int sizes[8] = {10, 100, 200, 500, 800, 1000, 2000, 3000};
    Eigen::setNbThreads(6);

    std::ofstream out("measurements.csv");
    out<<"Resolution,CPU,GPU\n";
    for(int j = 0; j < 8; j++)
    {
        int size = sizes[j];
        

        // Create CUDA instances and habdles
        cudaError cudaStatus;
        cusolverStatus_t cusolverStatus ;
        cusolverDnHandle_t handle;
        cusolverStatus = cusolverDnCreate(&handle);

        /* Create data structures in Eigen. 
           Equation considered is Ax=b, with A = I, b and x are random
           In the end, the solution should be x = b.
        */
        
        Eigen::MatrixXf A(size, size);
        Eigen::VectorXf b(size);
        Eigen::VectorXf x(size);
        A = Eigen::MatrixXf::Identity(size, size);
        b = Eigen::VectorXf::Random(size);
        x = Eigen::VectorXf::Random(size);

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
        auto start = std::chrono::high_resolution_clock::now();
        cusolverStatus = cusolverDnSpotrf(handle, uplo, size, d_A.data().get(), size, d_Work.data().get(), Lwork, d_info);
        cusolverStatus = cusolverDnSpotrs(handle, uplo, size, 1, d_A.data().get(), size, d_b.data().get(), size, d_info);
        cudaStatus = cudaDeviceSynchronize();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration_gpu = (stop - start);

        // Solve on the CPU
        start = std::chrono::high_resolution_clock::now();
        Eigen::LDLT<Eigen::MatrixXf> ldlt(size);
        ldlt.compute(A);
        x = ldlt.solve(b);
        stop = std::chrono::high_resolution_clock::now();
        auto duration_cpu = (stop - start);

        // Check results
        vec_b = d_b;
        Eigen::Map<Eigen::VectorXf> x_gpu(vec_b.data(), x.size());    

        std::cout<<"\nResolution : "<<size<<std::endl;
        std::cout<<"\tGPU (microseconds) : "<<std::chrono::duration_cast<std::chrono::milliseconds>(duration_gpu).count();
        std::cout<<"\n\tCPU : "<<std::chrono::duration_cast<std::chrono::milliseconds>(duration_cpu).count();
        std::cout<<"\n\tDifference (in 2 Norm) (milliseconds) : "<<(x - x_gpu).squaredNorm()<<std::endl;

        // Write measurements to file
        out<<size<<","<<std::chrono::duration_cast<std::chrono::milliseconds>(duration_cpu).count()<<","<<std::chrono::duration_cast<std::chrono::milliseconds>(duration_gpu).count()<<"\n";
    }

    out.close();    
    return 0;
}