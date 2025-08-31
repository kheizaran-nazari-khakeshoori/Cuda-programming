#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <chrono>
#include <cublas_v2.h>
#include <omp.h>


struct MatrixMultiplyFunctor
{
    const float *matrix1, *matrix2;
    float *result;
    int rowsMatrix1, colsMatrix1, colsMatrix2;

    MatrixMultiplyFunctor(const float *matrix1_, const float *matrix2_, float *result_, int rowsMatrix1_, int colsMatrix1_, int colsMatrix2_)
        : matrix1(matrix1_), matrix2(matrix2_), result(result_), rowsMatrix1(rowsMatrix1_), colsMatrix1(colsMatrix1_), colsMatrix2(colsMatrix2_) {}

    __device__ float operator()(int index) const
    {
        int row = index / colsMatrix2;
        int col = index % colsMatrix2;
        float sum = 0.0f;
        for (int k = 0; k < colsMatrix1; ++k)
        {
            sum += matrix1[row * colsMatrix1 + k] * matrix2[k * colsMatrix2 + col];
        }
        return sum;
    }

   
    void operator()(bool run_on_cpu)
    {
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < rowsMatrix1; ++row)
        {
            for (int col = 0; col < colsMatrix2; ++col)
            {
                float sum = 0.0f;
                for (int k = 0; k < colsMatrix1; ++k)
                {
                    sum += matrix1[row * colsMatrix1 + k] * matrix2[k * colsMatrix2 + col];
                }
                result[row * colsMatrix2 + col] = sum;
            }
        }
    }

    
    void operator()(cublasHandle_t handle)
    {
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    colsMatrix2, rowsMatrix1, colsMatrix1,
                    &alpha,
                    matrix2, colsMatrix2,  
                    matrix1, colsMatrix1,  
                    &beta,
                    result, colsMatrix2); 
    }
};

int main()
{
    int rowsMatrix1 = 500, colsMatrix1 = 500, colsMatrix2 = 500;

    
    thrust::host_vector<float> matrix1(rowsMatrix1 * colsMatrix1);
    thrust::host_vector<float> matrix2(colsMatrix1 * colsMatrix2);
    thrust::host_vector<float> result(rowsMatrix1 * colsMatrix2, 0);

    
    for (int i = 0; i < rowsMatrix1 * colsMatrix1; ++i)
        matrix1[i] = i % 10 + 1;
    for (int i = 0; i < colsMatrix1 * colsMatrix2; ++i)
        matrix2[i] = (i % 10 + 1) * 0.5f;

   
    auto start_cpu = std::chrono::high_resolution_clock::now();
    MatrixMultiplyFunctor cpu_functor(matrix1.data(), matrix2.data(), result.data(), rowsMatrix1, colsMatrix1, colsMatrix2);
    cpu_functor(true); 
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;

    
    thrust::device_vector<float> d_matrix1(matrix1.begin(), matrix1.end());
    thrust::device_vector<float> d_matrix2(matrix2.begin(), matrix2.end());
    thrust::device_vector<float> d_result(rowsMatrix1 * colsMatrix2, 0);
    

    auto start_gpu = std::chrono::high_resolution_clock::now();
    thrust::counting_iterator<int> indices(0);
    thrust::transform(indices, indices + (rowsMatrix1 * colsMatrix2), d_result.begin(),
                      MatrixMultiplyFunctor(thrust::raw_pointer_cast(d_matrix1.data()),
                                            thrust::raw_pointer_cast(d_matrix2.data()),
                                            thrust::raw_pointer_cast(d_result.data()),
                                            rowsMatrix1, colsMatrix1, colsMatrix2));
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_gpu = end_gpu - start_gpu;

    
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_matrix1_cublas, *d_matrix2_cublas, *d_result_cublas;
    cudaMalloc(&d_matrix1_cublas, rowsMatrix1 * colsMatrix1 * sizeof(float));
    cudaMalloc(&d_matrix2_cublas, colsMatrix1 * colsMatrix2 * sizeof(float));
    cudaMalloc(&d_result_cublas, rowsMatrix1 * colsMatrix2 * sizeof(float));

    cudaMemcpy(d_matrix1_cublas, matrix1.data(), rowsMatrix1 * colsMatrix1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2_cublas, matrix2.data(), colsMatrix1 * colsMatrix2 * sizeof(float), cudaMemcpyHostToDevice);

    auto start_cublas = std::chrono::high_resolution_clock::now();
    MatrixMultiplyFunctor cublas_functor(d_matrix1_cublas, d_matrix2_cublas, d_result_cublas, rowsMatrix1, colsMatrix1, colsMatrix2);
    cublas_functor(handle);
    auto end_cublas = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cublas = end_cublas - start_cublas;

    cublasDestroy(handle);
    cudaFree(d_matrix1_cublas);
    cudaFree(d_matrix2_cublas);
    cudaFree(d_result_cublas);

    
    std::cout << "CPU time (OpenMP Parallelized): " << duration_cpu.count() << " seconds" << std::endl;
    std::cout << "GPU Thrust time: " << duration_gpu.count() << " seconds" << std::endl;
    std::cout << "GPU cuBLAS time: " << duration_cublas.count() << " seconds" << std::endl;

    return 0;
}
