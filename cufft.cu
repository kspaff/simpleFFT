#include <iostream>
#include <stddef.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

// Macro to catch CUDA errors
#define CUDA_SAFE_CALL( call) do {                                             \
    cudaError err = call;                                                      \
    if (cudaSuccess != err) {                                                  \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",          \
                __FILE__, __LINE__, cudaGetErrorString( err) );                \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// Macro to catch cufft errors
#define CUFFT_SAFE_CALL( call) do {                                            \
    cufftResult err = call;                                                    \
    if (err != CUFFT_SUCCESS) {                                                \
        fprintf(stderr, "Cufft error in file '%s' in line %i : %s.\n",         \
                __FILE__, __LINE__, "error" );                                 \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

/*
   typedef enum cufftResult_t {
   CUFFT_SUCCESS        = 0x0,
   CUFFT_INVALID_PLAN   = 0x1,
   CUFFT_ALLOC_FAILED   = 0x2,
   CUFFT_INVALID_TYPE   = 0x3,
   CUFFT_INVALID_VALUE  = 0x4,
   CUFFT_INTERNAL_ERROR = 0x5,
   CUFFT_EXEC_FAILED    = 0x6,
   CUFFT_SETUP_FAILED   = 0x7,
   CUFFT_INVALID_SIZE   = 0x8,
   CUFFT_UNALIGNED_DATA = 0x9
   } cufftResult;
 */

using namespace std;

int main(int argc, char* argv[]) 
{
    if (!(argc == 2 || argc == 3)) { 
        cerr << "usage: ./fft N D, where N*1024 is number of elems, "
            << "and D is device number" << endl;
        exit(-1);
    }
    if (argc == 3) {
        cudaSetDevice(atoi(argv[2]));
    }

    int count = atoi(argv[1]) * 1024;
    size_t bytes = count * sizeof(cufftDoubleComplex);

    // Allocate host memory for the signal
    cufftDoubleComplex* h_signal = new cufftDoubleComplex[count];

    // Initalize the memory for the signal
    for (unsigned int i = 0; i < count; i++) {
        h_signal[i].x = 1.;
        h_signal[i].y = 0.;
    }

    // Allocate device memory for signal
    cufftDoubleComplex* d_signal;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_signal, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_signal, h_signal, bytes,
                cudaMemcpyHostToDevice));

    // CUFFT plan
    cufftHandle plan;
    CUFFT_SAFE_CALL(cufftPlan1d(&plan, count, CUFFT_Z2Z, 1));

    // Transform signal -- warm up
    CUFFT_SAFE_CALL(cufftExecZ2Z(plan, d_signal, d_signal, CUFFT_FORWARD));

    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));
    for (int i = 0; i < 100; i++) {
        CUFFT_SAFE_CALL(cufftExecZ2Z(plan, d_signal, d_signal, CUFFT_FORWARD));
    }
    cudaEventRecord(stop, 0);
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    elapsedTime *= 1.e-3;
    cout << count << ", " << elapsedTime / 100.0f << endl;

    // Copy device memory to host
    cufftDoubleComplex* h_convolved_signal = h_signal;
    CUDA_SAFE_CALL(cudaMemcpy(h_convolved_signal, d_signal, bytes, 
                cudaMemcpyDeviceToHost));


    //Destroy CUFFT context
    CUFFT_SAFE_CALL(cufftDestroy(plan));

    // cleanup memory
    delete[] h_signal;
    CUDA_SAFE_CALL(cudaFree(d_signal));
}
