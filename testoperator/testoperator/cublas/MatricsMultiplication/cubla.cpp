

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include "MatrixMultiCuBLAS.h"


#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */


template <typename T_ELEM>
void mat_fillupMatrix(T_ELEM *A , int lda , int rows, int cols, int seed)
{
    for (int j = 0; j < cols; j++)
    {
        for (int i = 0; i < rows; i++)
        {
            A[i + lda*j ] = cuGet<T_ELEM> (((double)(((lda*i+j+seed) % 253)+1))/256.0, ((double)((((cols*i+j) + 123 + seed) % 253)+1))/256.0);
        }
    }
}


static inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                         cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                         float *alpha, const float *A, int lda,
                                         float *B, int ldb, float *beta,
                                         float *C, int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

static inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                         cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                         double *alpha, const double *A, int lda,
                                         double *B, int ldb, double *beta,
                                         double *C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

static inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                                cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                                float *alpha, const float *Aarray[], int lda,
                                                const float *Barray[], int ldb, float *beta,
                                                float *Carray[], int ldc, int batchCount)
{
#if CUDART_VERSION >= 4010
    return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
    return CUBLAS_STATUS_SUCCESS;
#endif
}

static inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                                cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                                double *alpha, const double *Aarray[], int lda,
                                                const double *Barray[], int ldb, double *beta,
                                                double *Carray[], int ldc,
                                                int batchCount)
{
#if CUDART_VERSION >= 4010
    return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
#else
    return CUBLAS_STATUS_SUCCESS;
#endif
}






template <typename T_ELEM>
static int TESTGEN(gemm)(const struct gemmOpts *opts,
                         int matrixM, int matrixN, int matrixK, int &numTests,
                         struct gemmTestParams<T_ELEM> *params)
{
    static T_ELEM alpha[] = { cuGet<T_ELEM>(0,0), cuGet<T_ELEM>(-1,-1), cuGet<T_ELEM>(1,-2), cuGet<T_ELEM>(2,-1), cuGet<T_ELEM>(0,-3) };
    static T_ELEM beta[]  = { cuGet<T_ELEM>(0,0), cuGet<T_ELEM>(-1,-1), cuGet<T_ELEM>(1,-2), cuGet<T_ELEM>(2,-1), cuGet<T_ELEM>(0,-3)};

#define NBR_ALPHAS (sizeof(alpha) / sizeof(alpha[0]))
#define NBR_BETAS (sizeof(beta) / sizeof(beta[0]))
    static T_ELEM theAlpha;
    static T_ELEM theBeta;
    static int state;
    static int m;
    static int n;
    static int k;

    if (numTests-- <= 0)
    {
        return -1;
    }

    theAlpha = alpha[cuRand()%NBR_ALPHAS];
    theBeta  = beta[cuRand()%NBR_BETAS];
    params->transa = CUBLAS_OP_N;
    params->transb = CUBLAS_OP_N;
    m = matrixM;
    n = matrixN;
    k = matrixK;
    params->m = m;
    params->n = n;
    params->k = k;
    params->alpha = theAlpha;
    params->beta = theBeta;

    printf("\n");
    printf("MAtrix multiplicaton");
    m = cuRand() % matrixM;
    n = cuRand() % matrixN;
    k = cuRand() % matrixK;

    state = cuRand() % 9;
    return 0;
}






template <typename T_ELEM>
int test_gemm_loop(struct gemmOpts &opts, float err, double max_relative_error, cublasHandle_t handle)
{
    struct gemmTestParams<T_ELEM> params;
    cudaStream_t *streamArray = 0;
    cublasStatus_t status1, status2, status3;
    T_ELEM *A = NULL;
    T_ELEM *B = NULL;
    T_ELEM *C = NULL;
    T_ELEM **devPtrA = 0;
    T_ELEM **devPtrB = 0;
    T_ELEM **devPtrC = 0;
    T_ELEM **devPtrA_dev = NULL;
    T_ELEM **devPtrB_dev = NULL;
    T_ELEM **devPtrC_dev = NULL;
    int matrixM, matrixN, matrixK;
    int rowsA, rowsB, rowsC;
    int colsA, colsB, colsC;
    int matrixSizeA, matrixSizeB, matrixSizeC;
    int errors;
    double start, stop;

    printf("Testing %cgemm\n", *opts.elem_type);

    matrixM = (opts.m) ? opts.m : BENCH_MATRIX_M;
    matrixN = (opts.n) ? opts.n : BENCH_MATRIX_N;
    matrixK = (opts.k) ? opts.k : BENCH_MATRIX_K;

    rowsA = imax(1, matrixM);
    colsA = imax(1, matrixK);
    rowsB = imax(1, matrixK);
    colsB = imax(1, matrixN);
    rowsC = imax(1, matrixM);
    colsC = imax(1, matrixN);

    matrixSizeA = rowsA * colsA;
    matrixSizeB = rowsB * colsB;
    matrixSizeC = rowsC * colsC;

    devPtrA =(T_ELEM **)malloc(opts.N * sizeof(*devPtrA));
    devPtrB =(T_ELEM **)malloc(opts.N * sizeof(*devPtrB));
    devPtrC =(T_ELEM **)malloc(opts.N * sizeof(*devPtrC));

    for (int i = 0; i < opts.N ; i++)
    {
        cudaError_t err1 = cudaMalloc((void **)&devPtrA[i], matrixSizeA * sizeof(devPtrA[0][0]));
        cudaError_t err2 = cudaMalloc((void **)&devPtrB[i], matrixSizeB * sizeof(devPtrB[0][0]));
        cudaError_t err3 = cudaMalloc((void **)&devPtrC[i], matrixSizeC * sizeof(devPtrC[0][0]));

        if ((err1 != cudaSuccess) ||
            (err2 != cudaSuccess) ||
            (err3 != cudaSuccess))
        {
            CLEANUP();
            fprintf(stderr, "!!!! GPU memory allocation error\n");
            return CUBLASTEST_FAILED;
        }
    }

    // For batched processing we need those arrays on the device
    if (opts.test_method == tmBatched)
    {
        cudaError_t err1 = cudaMalloc((void **)&devPtrA_dev, opts.N * sizeof(*devPtrA));
        cudaError_t err2 = cudaMalloc((void **)&devPtrB_dev, opts.N * sizeof(*devPtrB));
        cudaError_t err3 = cudaMalloc((void **)&devPtrC_dev, opts.N * sizeof(*devPtrC));

        if ((err1 != cudaSuccess) ||
            (err2 != cudaSuccess) ||
            (err3 != cudaSuccess))
        {
            CLEANUP();
            fprintf(stderr, "!!!! GPU memory allocation error\n");
            return CUBLASTEST_FAILED;
        }

        err1 = cudaMemcpy(devPtrA_dev, devPtrA, opts.N * sizeof(*devPtrA), cudaMemcpyHostToDevice);
        err2 = cudaMemcpy(devPtrB_dev, devPtrB, opts.N * sizeof(*devPtrB), cudaMemcpyHostToDevice);
        err3 = cudaMemcpy(devPtrC_dev, devPtrC, opts.N * sizeof(*devPtrC), cudaMemcpyHostToDevice);

        if ((err1 != cudaSuccess) ||
            (err2 != cudaSuccess) ||
            (err3 != cudaSuccess))
        {
            CLEANUP();
            fprintf(stderr, "!!!! cannot copy pointer array to device\n");
            return CUBLASTEST_FAILED;
        }
    }

    A  = (T_ELEM *)malloc(matrixSizeA * sizeof(A[0]));
    B  = (T_ELEM *)malloc(matrixSizeB * sizeof(B[0]));
    C  = (T_ELEM *)malloc(matrixSizeC * sizeof(C[0]));

    if ((!A) || (!B) || (!C))
    {
        CLEANUP();
        fprintf(stderr, "!!!! system memory allocation error\n");
        return CUBLASTEST_FAILED;
    }

    streamArray = (cudaStream_t *)malloc(opts.N * sizeof(cudaStream_t *));

    for (int i = 0; i < opts.N ; i++)
    {
        if (opts.test_method == tmStream)
        {
            cudaError_t cudaErr = cudaStreamCreate(&streamArray[i]);

            if (cudaErr != cudaSuccess)
            {
                CLEANUP();
                fprintf(stderr, "!!!! cannot create stream\n");
                return CUBLASTEST_FAILED;
            }
        }
        else
        {
            streamArray[i] = 0;
        }
    }

    errors = 0;
    int numTests = 1;

    while (TESTGEN(gemm)(&opts, matrixM, matrixN, matrixK, numTests, &params) == 0)
    {

        // fillup with Nan first (so lda padding is full on Nan)
        memset(A, 0xFF, matrixSizeA* sizeof(A[0]));
        mat_fillupMatrixDebug(A, rowsA, params.m, params.k);
        memset(B, 0xFF, matrixSizeB* sizeof(B[0]));
        mat_fillupMatrix(B, rowsB, params.k, params.n, 121);

        if (!cuEqual(params.beta, cuGet<T_ELEM>(0)))
        {
            mat_fillupMatrix(C, rowsC, params.m, params.n);
        }
        else
        {
            /* fill with SNaNs to make sure ZGEMM doesn't access C */
            memset(C, 0xFF, matrixSizeC * sizeof(C[0]));
        }

        double flopsCoef = 2.0;

        for (int i = 0; i < opts.N ; i++)
        {
            status1 = cublasSetMatrix(rowsA, colsA, sizeof(A[0]), A, rowsA, devPtrA[i], rowsA);
            status2 = cublasSetMatrix(rowsB, colsB, sizeof(B[0]), B, rowsB, devPtrB[i], rowsB);
            status3 = cublasSetMatrix(rowsC, colsC, sizeof(C[0]), C, rowsC, devPtrC[i], rowsC);

            if ((status1 != CUBLAS_STATUS_SUCCESS) || (status2 != status1) || (status3 != status1))
            {
                CLEANUP();
                fprintf(stderr, "!!!! GPU access error (write)\n");
                return CUBLASTEST_FAILED;
            }
        }

        start = second();

        if (opts.test_method == tmBatched)
        {
            cublasSetStream(handle, streamArray[0]);
            status1 = cublasXgemmBatched(handle, params.transa, params.transb, params.m, params.n,
                                         params.k, &params.alpha, (const T_ELEM **) devPtrA_dev, rowsA,
                                         (const T_ELEM **) devPtrB_dev, rowsB, &params.beta, devPtrC_dev, rowsC, opts.N);

            if (status1 != CUBLAS_STATUS_SUCCESS)
            {
                cudaError_t cudaStatus = cudaGetLastError();
                CLEANUP();
                fprintf(stderr, "!!!! GPU program execution error : cublas Error=%d, cuda Error=%d,(%s)\n", status1, cudaStatus,cudaGetErrorString(cudaStatus));
                return CUBLASTEST_FAILED;
            }
        }
        else
        {
            for (int i = 0; i < opts.N ; i++)
            {
                cublasSetStream(handle, streamArray[i]);
                status1 = cublasXgemm(handle, params.transa, params.transb, params.m, params.n,
                                      params.k, &params.alpha, devPtrA[i], rowsA,
                                      devPtrB[i], rowsB, &params.beta, devPtrC[i], rowsC);

                if (status1 != CUBLAS_STATUS_SUCCESS)
                {
                    cudaError_t cudaStatus = cudaGetLastError();
                    CLEANUP();
                    fprintf(stderr, "!!!! GPU program execution error : cublas Error=%d, cuda Error=%d,(%s)\n", status1, cudaStatus,cudaGetErrorString(cudaStatus));
                    return CUBLASTEST_FAILED;
                }
            }
        }

        cudaError_t cudaStatus = cudaDeviceSynchronize();

        if (cudaStatus != cudaSuccess)
        {
            CLEANUP();
            fprintf(stderr, "!!!! GPU program execution error on cudaDeviceSynchronize : cudaError=%d,(%s)\n", cudaStatus,cudaGetErrorString(cudaStatus));
            return CUBLASTEST_FAILED;
        }

        stop = second();

        fprintf(stdout, "^^^^ Latancy = %10.8f sec Throughput(GFLOPS)=%g\n", (stop-start),
                opts.N * (1e-9*flopsCoef*params.m*params.n*params.k)/(stop-start));

    } // end while (TESTGEN..

    CLEANUP();
    fprintf(stdout, "@@@@ %cgemm test %s\n", *opts.elem_type ,errors ? "FAIL" : "OK");
    return CUBLASTEST_PASSED;
}





int main(int argc, char** argv)
{    
    //CAI_AG - Reading values for input parameters using command line arguments 
    for (int i = 0;i <= 5; i++)
        std::cout << argv[i] << std::endl;
    int n, c, h, w;
    std::string a;
    for (int i = 1; i <= 5; i++) {
        int len = sizeof(argv[i]);
        if (argv[i][1] == 'n')
          n = atoi(argv[i] + 2);
        else if (argv[i][1] == 'c')
          c = atoi(argv[i] + 2);
        else if (argv[i][1] == 'h')
          h = atoi(argv[i] + 2);
        else if (argv[i][1] == 'w')
          w = atoi(argv[i] + 2);
        else if (argv[i][1] == 'a')
          a = argv[i] + 2; 
   }

    //CAI_AG - Generating random input_data 
    int size = n*c*h*w;
    int input_data[size];
    for (int i = 0; i < size; i++)
      input_data[i] = rand() % 10;
 
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;
    cudaSetDevice(0); // use GPU0
    int device; 
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;

    
    struct gemmOpts opts;
    int errors, nTimes, nTotalErrors = 0;
    int status = CUBLASTEST_PASSED;

    
    cublasHandle_t handle;

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stdout, "CUBLAS initialization failed!\n");
        exit(EXIT_FAILURE);
    }

    // Run single kernels
    fprintf(stdout, "\n ==== Running single kernels ==== \n\n");
    nTimes = opts.N;
    opts.N = 1;
    *(opts.elem_type) = 's';
    status = test_gemm_loop<float>(opts, (float)CUBLAS_SGEMM_MAX_ULP_ERR, (double)CUBLAS_SGEMM_MAX_RELATIVE_ERR, handle);
    
 
   

     // Run Double version
    *(opts.elem_type) = 'd';

    if (getDeviceVersion() < DEV_VER_DBL_SUPPORT)
    {
        fprintf(stdout, "@@@@ dgemm test WAIVED due to lack of DP support\n");
        exit(EXIT_WAIVED);
    }

    status = test_gemm_loop<double>(opts, (float)CUBLAS_DGEMM_MAX_ULP_ERR, (double)CUBLAS_DGEMM_MAX_RELATIVE_ERR, handle);
    nTotalErrors += (status == CUBLASTEST_PASSED ? 0 : 1);
    opts.N = nTimes;    
       
     
    #if CUDART_VERSION >= 4010

    for (int ii = 0; ii < 3; ii++)
    {
#else

    for (int ii = 0; ii < 2; ii++)
    {
#endif

        switch (ii)
        {
            case 0:
                opts.test_method = tmRegular;
                fprintf(stdout, "\n ==== Running N=%d without streams ==== \n\n", opts.N);
                break;

            case 1:
                opts.test_method = tmStream;
                fprintf(stdout, "\n ==== Running N=%d with streams ==== \n\n", opts.N);
                break;

            case 2:
                opts.test_method = tmBatched;
                fprintf(stdout, "\n ==== Running N=%d batched ==== \n\n", opts.N);
                break;
        }

        // Run single version
        *(opts.elem_type) = 's';
        status = test_gemm_loop<float>(opts, (float)CUBLAS_SGEMM_MAX_ULP_ERR, (double)CUBLAS_SGEMM_MAX_RELATIVE_ERR, handle);
        nTotalErrors += (status == CUBLASTEST_PASSED ? 0 : 1);

        // Run Double version
        *(opts.elem_type) = 'd';

        // Test doesn't meet minSpec, will will wave the DP test
        if (getDeviceVersion() < DEV_VER_DBL_SUPPORT)
        {
            fprintf(stdout, "@@@@ dgemm test WAIVED due to lack of DP support\n");
            exit(EXIT_WAIVED);
        }
        else
        {
            status = test_gemm_loop<double>(opts, (float)CUBLAS_DGEMM_MAX_ULP_ERR, (double)CUBLAS_DGEMM_MAX_RELATIVE_ERR, handle);
            nTotalErrors += (status == CUBLASTEST_PASSED ? 0 : 1);
        }
    }

    cublasDestroy(handle);

    printf("\nTest Summary\n");
    printf("%d error(s)\n", nTotalErrors);
    



}




