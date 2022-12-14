////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// Matrix multiplication: C = A * B.
// Host code.
//
// This sample implements matrix multiplication as described in Chapter 3
// of the programming guide and uses the CUBLAS library to demonstrate
// the best performance.

// SOME PRECAUTIONS:
// IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
// WE JUST NEED CALL CUBLAS API IN A REVERSE ORDER: cublasSegemm(B, A)!
// The reason is explained as follows:

// CUBLAS library uses column-major storage, but C/C++ use row-major storage.
// When passing the matrix pointer to CUBLAS, the memory layout alters from
// row-major to column-major, which is equivalent to an implicit transpose.

// In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
// C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
// implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
// If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
// multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
// is a column-based cublas matrix, which means C(T) in C/C++, we need extra
// transpose code to convert it to a row-based C/C++ matrix.

// To solve the problem, let's consider our desired result C, a row-major matrix.
// In cublas format, it is C(T) actually (because of the implicit transpose).
// C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
// happen to be C/C++ matrice B and A (still because of the implicit transpose)!
// We don't need extra transpose code, we only need alter the input order!
//
// CUBLAS provides high-performance matrix multiplication.
// See also:
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#include <nvml.h>

#include <chrono>
using namespace std::chrono;
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include <string>

using namespace std;
using std::cout;
using std::endl;

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

int start_measure_cpu() {
    pid_t  pid;
    int    status;
    char * cmd[2];
    cmd[0]="cpu-energy-meter";
    cmd[1]=NULL;

    if ((pid = fork()) < 0) {     /* fork a child process           */
        printf("*** ERROR: forking child process failed\n");
        exit(1);
    }
    else if (pid == 0) {          /* for the child process:         */
        if (execvp("/home/jieyang/software/cpu-energy-meter/cpu-energy-meter", cmd) < 0) {     /* execute the command  */
            printf("*** ERROR: exec failed\n");
            exit(1);
        }
    }
    return pid;
}

void stop_measure_cpu(int pid) {
    string pid_str = std::to_string(pid);
    string kill_cmd = "sudo kill -2 "+ pid_str;
    printf("executing: %s\n", kill_cmd.c_str());
    system(kill_cmd.c_str());
}

unsigned long long start_measure_gpu(nvmlDevice_t device) {
    unsigned long long energy;
    nvmlDeviceGetTotalEnergyConsumption(device, &energy);
    return energy;
}

unsigned long long stop_measure_gpu(nvmlDevice_t device, unsigned long long start_energy) {
    unsigned long long stop_energy;
    nvmlDeviceGetTotalEnergyConsumption(device, &stop_energy);
    return stop_energy - start_energy;
    //printf("GPU energy: %llu\n", stop_energy - start_energy);
}


void change_offset(int offset){

    string offset_str = std::to_string(offset);
    string offset_cmd = "sudo nvidia-settings -a [gpu:0]/GPUGraphicsClockOffset[4]="+ offset_str;
    system(offset_cmd.c_str());
}

int undervolte(nvmlDevice_t device, int clock, int power)
{
   

    nvmlReturn_t result;


    result = nvmlDeviceSetPersistenceMode(device, NVML_FEATURE_ENABLED);
    if (NVML_SUCCESS != result)
    {
        printf("Failed to set PM: %s.\n", nvmlErrorString(result));
        return -1;
    }

    result = nvmlDeviceSetPowerManagementLimit (device, power);
    if (NVML_SUCCESS != result)
    {
      printf("Failed to set power limit of device: %s\n", nvmlErrorString(result));
      return -1;
    }
    // unsigned int get_power_limit;
    // result = nvmlDeviceGetEnforcedPowerLimit(device, &get_power_limit);
    // if (NVML_SUCCESS != result) {
    //     printf("Failed to get power limit of device %i: %s\n", i, nvmlErrorString(result));
    //     return;
    // }
    // printf("Power limit set to: %d\n", get_power_limit);


    result = nvmlDeviceSetGpuLockedClocks ( device, clock, clock );
    if (NVML_SUCCESS != result)
    {
      printf("Failed to set clock of device: %s\n", nvmlErrorString(result));
      return -1;
    }
    unsigned int get_gpu_clock;
    result = nvmlDeviceGetClock(device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, &get_gpu_clock);
    if (NVML_SUCCESS != result)
    {
      printf("Failed to get GPU clock of device: %s\n", nvmlErrorString(result));
      return -1;
    }
    unsigned int get_mem_clock;
    result = nvmlDeviceGetClock(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &get_mem_clock);
    if (NVML_SUCCESS != result)
    {
      printf("Failed to get memory clock of device: %s\n", nvmlErrorString(result));
      return -1;
    }
    //printf("Clock is set to: %u, %u\n", get_gpu_clock, get_mem_clock);
    return get_gpu_clock;

    // nvmlEnableState_t set = NVML_FEATURE_DISABLED;
    // result = nvmlDeviceSetAutoBoostedClocksEnabled(device, set);
    // if (NVML_SUCCESS != result)
    // {
    //   printf("Failed to disable autoboost of device %i: %s\n", i, nvmlErrorString(result));
    //   return;
    // }

}


////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void
matrixMulCPU(T *C, const T *A, const T *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            T sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                T a = A[i * wA + k];
                T b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (T)sum;
        }
}

// Allocates a matrix with random double entries.
template <typename T>
void randomInit(T *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (T)RAND_MAX;
}


template <typename T>
void printDiff(T *data1, T *data2, int width, int height, int iListLength, T fListTol)
{
    //printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            //printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            T fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    //printf(" \n  Total Errors = %d\n", error_count);
}



////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
template <typename T>
int matrixMultiply(int n)
{
    if (nvmlInit () != NVML_SUCCESS)
    {
        cout << "init error";
        return 0;
    }
    int i = 0;
    nvmlReturn_t result;
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(i, &device);
    if (NVML_SUCCESS != result)
    {
      printf("Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));
      return 0;
    }

    change_offset(0);

    //int measure_cpu_pid = start_measure_cpu();

    sMatrixSize matrix_size;
    //int n = 51200-1024*5;//30720;
    //int n = 30720;

    int b = 256;
    // undervolte(device, 1800, 100000);

    // allocate host memory for matrices A and B
    unsigned int size_A = n * b;
    size_t mem_size_A = sizeof(T) * size_A;
    T *h_A = (T *)malloc(mem_size_A);

    unsigned int size_B = b * n;
    size_t mem_size_B = sizeof(T) * size_B;
    T *h_B = (T *)malloc(mem_size_B);

    unsigned int size_C = n * n;
    size_t mem_size_C = sizeof(T) * size_C;
    T *h_C      = (T *) malloc(mem_size_C);
    T *h_C2      = (T *) malloc(mem_size_C);

    T *d_A, *d_B, *d_C;//, *d_C2;
    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));
    printf("allocate: %llu\n", mem_size_A+mem_size_B+mem_size_C);
    //checkCudaErrors(cudaMalloc((void **) &d_C2, mem_size_C));
    //d_C2 = d_C;
    // set seed for rand()
    srand(2006);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    const T alpha = 1.0f;
    const T beta  = 0.0f;
    cublasHandle_t handle;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    high_resolution_clock::time_point t1, t2;
    double nvml_time;
    checkCudaErrors(cublasCreate(&handle));
    unsigned long long start_energy;

    printf("Computing result using CUBLAS (normal power)...\n");
    undervolte(device, 1500, 338000);
    checkCudaErrors(cudaEventRecord(start, NULL));
    if (sizeof(T) == 8) {
        checkCudaErrors(cublasDgemm(handle, 
                                    CUBLAS_OP_N, CUBLAS_OP_N, 
                                    n, n, b, 
                                    (double*)&alpha, 
                                    (double*)d_A, n, 
                                    (double*)d_B, b, 
                                    (double*)&beta, 
                                    (double*)d_C, n));
    } else {
        checkCudaErrors(cublasSgemm(handle, 
                                    CUBLAS_OP_N, CUBLAS_OP_N, 
                                    n, n, b, 
                                    (float*)&alpha, 
                                    (float*)d_A, n, 
                                    (float*)d_B, b, 
                                    (float*)&beta, 
                                    (float*)d_C, n));
    }

    checkCudaErrors(cudaMemcpy(h_C2, d_C, mem_size_C, cudaMemcpyDeviceToHost));

   
    // create and start timer
    printf("Computing result using CUBLAS (low power)...\n");


    // t1 = high_resolution_clock::now(); 
    // system("sudo ls > /dev/null");
    // t2 = high_resolution_clock::now();
    // nvml_time = duration_cast<milliseconds>(t2 - t1).count();
    // printf("CPUPOWER time: %f\n", nvml_time);


    // t1 = high_resolution_clock::now(); 
    // system("sudo cpupower frequency-set -u 2000000 > /dev/null");
    // t2 = high_resolution_clock::now();
    // nvml_time = duration_cast<milliseconds>(t2 - t1).count();
    // printf("CPUPOWER time: %f\n", nvml_time);

    // t1 = high_resolution_clock::now(); 
    // system("sudo cpupower frequency-set -u 3000000 > /dev/null");
    // t2 = high_resolution_clock::now();
    // nvml_time = duration_cast<milliseconds>(t2 - t1).count();
    // printf("CPUPOWER time: %f\n", nvml_time);


    // t1 = high_resolution_clock::now(); 
    // system("sudo cpupower frequency-set -u 4000000 > /dev/null");
    // t2 = high_resolution_clock::now();
    // nvml_time = duration_cast<milliseconds>(t2 - t1).count();
    // printf("CPUPOWER time: %f\n", nvml_time);

    // change_offset(0);
    // // execute the kernel
    // int actual_f = undervolte(device, 500, 100000);
    // unsigned int static_power_mill;
    // nvmlDeviceGetPowerUsage(device, &static_power_mill);
    // double static_power = static_power_mill / 1000.0;
    // printf("static: %f\n", static_power);

    // Record the start event
    //checkCudaErrors(cudaEventRecord(start, NULL));

    change_offset(200);

    int repeat = 100;
    int fail_count = 0;
    double total_perf = 0.0;
    for (int f = 1900; f <= 2000; f+=100)
    // f = 1600;
    {
        int actual_f = undervolte(device, f, 338000);
        undervolte(device, f, 338000);
        undervolte(device, f, 338000);
        undervolte(device, f, 338000);
        undervolte(device, f, 338000);
        undervolte(device, f, 338000);

        checkCudaErrors(cudaEventRecord(start, NULL));
        start_energy = start_measure_gpu(device);
    	//int measure_cpu_pid = start_measure_cpu();      
    	for (int r = 0; r < repeat; r++) {
            if (sizeof(T) == 8) {
                checkCudaErrors(cublasDgemm(handle, 
                                            CUBLAS_OP_N, CUBLAS_OP_N, 
                                            n, n, b, 
                                            (double*)&alpha, 
                                            (double*)d_A, n, 
                                            (double*)d_B, b, 
                                            (double*)&beta, 
                                            (double*)d_C, n));
            } else {
                checkCudaErrors(cublasSgemm(handle, 
                                            CUBLAS_OP_N, CUBLAS_OP_N, 
                                            n, n, b, 
                                            (float*)&alpha, 
                                            (float*)d_A, n, 
                                            (float*)d_B, b, 
                                            (float*)&beta, 
                                            (float*)d_C, n));
            }
            // checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
            // printDiff(h_C, h_C2, n, n, 10, (T)0.00001);
        }

	
        checkCudaErrors(cudaEventRecord(stop, NULL));
        checkCudaErrors(cudaEventSynchronize(stop));
        start_energy = stop_measure_gpu(device, start_energy);
        //stop_measure_cpu(measure_cpu_pid);
	float msecTotal = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
        //Compute and print the performance
        float msecPerMatrixMul = msecTotal;
	    

        

        float flopsPerMatrixMul = 2.0 * (double)n * (double)n * (double)b * repeat;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        double power = start_energy/msecPerMatrixMul;
        printf("%d MHz, %d MHz, %.3f ms, %.3f GFlop, %.3f W, %.3f GFlops/W\n",
             f, actual_f, msecPerMatrixMul, gigaFlops, power, gigaFlops/power);
	total_perf += gigaFlops;

        nvmlDeviceResetGpuLockedClocks(device);
    }
	// printf("total test: %d, failed: %d.\n", nIter, fail_count);
	// printf("failure rate: %f.\n", (double)fail_count/nIter);
	// printf("average perf: %.2f.\n", total_perf/nIter);
        // printf("done.\n");

        // // Record the stop event
        // checkCudaErrors(cudaEventRecord(stop, NULL));

        // // Wait for the stop event to complete
        // checkCudaErrors(cudaEventSynchronize(stop));

        // double msecTotal = 0.0f;
        // checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

        // // Compute and print the performance
        // double msecPerMatrixMul = msecTotal / nIter;
        // double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC * (double)matrix_size.uiWC * (double)matrix_size.uiHB;
        // double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        // printf(
        //     "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
        //     gigaFlops,
        //     msecPerMatrixMul,
        //     flopsPerMatrixMul);

        // copy result from device to host
        //checkCudaErrors(cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost));

        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));

    

    

    //printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    // clean up memory

    //stop_measure_cpu(measure_cpu_pid);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C2);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    undervolte(device, 1500, 338000);
    change_offset(0);
    //checkCudaErrors(cudaFree(d_C2));

    // if (resCUBLAS == true)
    // {
    //     return EXIT_SUCCESS;    // return value = 1
    // }
    // else
    // {
    //     return EXIT_FAILURE;     // return value = 0
    // }
    return 0;
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    
    int devID = 0, sizeMult = 5;
    sMatrixSize matrix_size;



    // matrixMultiply<double>(30720);
    // matrixMultiply<double>(25600);
    // matrixMultiply<double>(20480);
    // matrixMultiply<double>(15360);
    // matrixMultiply<double>(10240);
    // matrixMultiply<double>(5120);

    for (int n = 40960; n <= 40960; n+=5120) {
        printf("[Matrix Multiply CUBLAS %d] - Starting...\n", n);
        matrixMultiply<float>(n);
    }
    //resetvolte();
    return 0;
}
