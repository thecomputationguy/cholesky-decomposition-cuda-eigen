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

int main(int argc, char* argv[])
{
    std::cout<<"\n***** Starting Cholesky Solvers *****"<<std::endl;

    std::vector<int> resolutions = read_file();
    Eigen::setNbThreads(omp_get_max_threads()); // To enable multithreaded computation, if and when available.
    std::chrono::steady_clock::time_point start; // start timer
    std::chrono::steady_clock::time_point stop; // stop timer

    std::ofstream out("measurements.csv"); // to store the measurement data
    out<<"Resolution,CPU,GPU,GPU-Speedup\n";

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
        Equation considered is Ax=b, with A = I, b and x are random
        In the end, the solution should be x = b.
        */
        
        Eigen::MatrixXf A(size, size);
        Eigen::VectorXf b(size), x(size);
        A = Eigen::MatrixXf::Identity(size, size);
        b = x = Eigen::VectorXf::Random(size);

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

        std::cout<<"\nResolution : "<<size<<std::endl;
        std::cout<<"\tGPU (milliseconds) : "<<duration_gpu / 1000;
        std::cout<<"\n\tCPU (milliseconds) : "<<duration_cpu / 1000;
        std::cout<<"\n\tDifference (in 2 Norm) : "<<(x - x_gpu).squaredNorm()<<std::endl;
        std::cout<<"\n\tGPU Speedup : "<<speedup<<std::endl;

        // Write measurements to file
        out<<size<<","<<duration_cpu / 1000<<","<<duration_gpu / 1000<<","<<speedup<<"\n";
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
