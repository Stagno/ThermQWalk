#define THERMQWALK

#ifdef _MSC_VER
#define STRINGIZE_HELPER(x) #x
#define STRINGIZE(x) STRINGIZE_HELPER(x)
#define __MESSAGE(text) __pragma( message(__FILE__ "(" STRINGIZE(__LINE__) ")" text) )
#define WARNING(text) __MESSAGE( " : Warning: " #text )
#define ERROR(text) __MESSAGE( " : Error: " #text )
#define MESSAGE(text) __MESSAGE( ": " #text )
#define TODO(text) WARNING( TODO: text )
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math_constants.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>

#include "helper.h"
#include "input_parameters.cuh"

// real and complex types are used when high floating point precision might be required (depends on the choice of parameters)
// Set to float (cuFloatComplex) or double (cuDoubleComplex) to represent real and complex numbers
typedef double real;
typedef cuFloatComplex complex;

#define PI_DOUBLE 3.141592653589793

#define N_MODES (N_SITES-1) // number of spring / modes
#define N_SITES2 (N_SITES)*(N_SITES)
#define N_MODES2 (N_MODES)*(N_MODES)
#define HBAR 5.194037869154565e-29
#define BOLTZMANN 7.942974009171879e-32


#define FIRST_TARGET_NODE ((int) (N * (1.0-PERC_TARGET)))

#define SIM_DT (double(SIM_LENGTH)/(SIM_POINTS-1))

// Device dependent settings

#define ARCH_MAX_THREADS 1024

#define CHUNK_R 7680
#define R_CHUNKS ((R + CHUNK_R - 1) / CHUNK_R)

#define CHUNK_SIZE_K 1
#define CHUNK_SIZE_M 1
#define CHUNK_SIZE_T 21
#define CHUNK_SIZE_GAMMA 3

// Simulation settings

//#define SIM_VERBOSE   // Define when want to output the deviation from unity of the sq. mod.
// of the numerical estimate of |psi(t+dt)>. WARNING: very high glob. mem. usage

/* Macros */

#define CUDASYNC() handleCudaError(cudaDeviceSynchronize())

/* Data structures */

struct DATAPOINT {

	float k,T,gamma;
	real m;
	float perf_idx;
	real time_idx;

	bool operator< (const DATAPOINT& other) const {

		if(gamma < other.gamma )
			return true;
		else if(other.gamma == this->gamma && T < other.T )
				return true;
		else if(other.gamma == this->gamma && other.T == this->T && m < other.m )
				return true;
		else if(other.gamma == this->gamma && other.T == this->T && other.m == this->m && k < other.k )
				return true;

		return false;
	}
};

/* Complex numbers auxiliary functions */

__host__ __device__ static __inline__ cuFloatComplex cuCadd(cuFloatComplex x, cuFloatComplex y) {
	return cuCaddf(x, y);
}

__host__ __device__ static __inline__ cuFloatComplex cuCmul(cuFloatComplex x, cuFloatComplex y) {
	return cuCmulf(x, y);
}

// This function sets a float complex number given its real and imaginary parts as floats
template<typename T_A, typename T_B>
__host__ __device__ static __inline__ T_A cuCset(const T_B r, const T_B i) {
	T_A c;
	c.x = r;
	c.y = i;
	return c;
}

// This function scales a T_A complex number by a T_B type real number
template<typename T_A, typename T_B>
__host__ __device__ static __inline__ T_A cuCscale(const T_A a, const T_B b) {
	T_A c;
	c.x = a.x*b;
	c.y = a.y*b;
	return c;
}

// Constants

//	Return real unit
__device__ __host__ static __inline__ cuComplex realOne() { return cuCset<cuComplex, float>(1.0f, 0.0f); }

//	Differential costant: -i/hbar (hbar = 1 for our purposes)
__device__ __host__ static __inline__ cuComplex diffC() { return cuCset<cuComplex, float>(0.0f, -1.0f); }

// Device functions
template<int N>
__device__ constexpr real eigenvecNormFactor();

template<typename T, int stride, int size>
__device__ __forceinline__ void ZeroStridedVectors(const int glob_idx, T * const d_ptr);

template<int stride, int size>
__device__ __forceinline__ void ZeroStridedVectors(const int glob_idx, float2 * const d_ptr);

template<int stride, int size>
__device__ __forceinline__ void ZeroStridedVectors(const int glob_idx, double2 * const d_ptr);

template<typename T_A, typename T_B, int stride, int size>
__device__ __forceinline__ void AddScaledCVecToCVec(const int glob_idx, T_A * const a, const T_B b, T_A * const c,T_A * const out);

template<typename T_IN,typename T_OUT>
__device__ __forceinline__ T_OUT cuCSqrdAbs(const T_IN c);

template<typename T_IN,typename T_OUT, int stride, int size>
__device__ __forceinline__ T_OUT CVecSqrdAbs(const int glob_idx, T_IN * const vec);

template<typename T_IN,typename T_OUT, int stride, int size>
__device__ __forceinline__ T_OUT CVecNorm(const int glob_idx, T_IN * const vec);

template<typename T_C, typename T_R, int stride, int size>
__device__ __forceinline__ void CVecScale(const int glob_idx, T_C * const vec,const T_R scale, T_C * const out);

template<typename T, int stride, int size>
__device__ __forceinline__ void VecCopy(const int glob_idx, T * const in,T * const out);

template<int block_dim_x, int block_dim_y, int num_tot,int stride,int n, int N>
__device__ __forceinline__ void evalKappa(const int2 i, const int glob_idx, const real t, float * const d_A,real * const d_omega, float * const d_phi,const float gamma, real * const d_orthU, float * const d_osc, float * const d_out);

template<int block_dim_x, int block_dim_y,int num_tot,int stride,int n, int N>
__device__ __forceinline__ void diff(const int2 i, const int glob_idx, const real t, cuComplex * const d_psi,
																			float * const d_diag, float * const d_A,
																			real * const d_omega, float * const d_phi, const float gamma, real * const d_orthU,
																			float * const d_kappa, float * const d_osc, cuComplex * const d_out);

template<int block_dim_x, int block_dim_y,int num_tot,int stride, int n, int N>
__device__ __forceinline__ void rk4(const int2 i, const int glob_idx, const real t, const real dt,
																			cuComplex * const d_psi, float * const d_diag, float * const d_A,
																			real * const d_omega, float * const d_phi, const float gamma, real * const d_orthU,
																			float * const d_kappa, float * const d_osc, cuComplex * const d_K[4], cuComplex * const d_out);

template<int stride,int n, int N>
__device__ __forceinline__ float stateStdDev(const int glob_idx,
																										cuComplex * const d_psi);

template<int stride,int n, int N>
__device__ __forceinline__ float targetProbability(const int glob_idx,
																										cuComplex * const d_psi);

template<int num_k, int num_m, int num_T, int num_gamma, int num_tot>
__global__ void FindMaxPerfIdx(float * const k_array, real * const m_array,
																		float * const T_array, float * const gamma_array,
																		float * const d_avg_perf_idx, DATAPOINT * const out);

template<int num_tot,int threadsPerBlock>
__global__ void AveragePerfIdx(float * const d_perf_idx, float * const d_out);

template<int block_dim_x, int block_dim_y, int stride_x, int stride_y, int num_k, int num_m, int num_T, int num_gamma, int num_tot,int n, int N>
__global__ void RunSim(float * const k_array, real * const m_array,
												float * const T_array, float * const gamma_array,
												cuComplex * const d_psi, float * const d_diag, float * const d_A,
												real * const d_omega, float * const d_phi, real * const d_orthU,
												float * const d_kappa, float * const d_osc, cuComplex * const d_K[4], cuComplex * const d_psi_next,
												float * const d_perf_idx /* vector of num_tot * CHUNK_R * SIM_POINTS performance indexes */,
												float * const d_deviations /* (optional) deviation from 1 of norm of est. |psi(t+dt)> */);

template<int N>
__global__ void SampleDisorderVec(const float disorder_param, float * const d_diag, curandState * const states);

template<int stride_x, int stride_y, int num_k, int num_m, int num_T, int num_gamma, int num_tot, int n, int N>
__global__ void SampleAmplitudesAndPhases(float * const k_array, real * const m_array,
														float * const T_array,
														float * const d_A, real * const d_omega, float * d_phi,
														curandState * const states);

__global__ void InitRNG(curandState *states, int seed, int num_gen);

template<int n, int N, int n2>
__global__ void StoreOrthU(real * const d_ptr);

template<typename T>
__global__  void VecInit(T * const d_ptr, const T val, const int D);
