#include <chrono>
#include <iostream>
#include <iomanip>

#include "ThermQWalk.cuh"
#include "output_utility.cuh"

/* Device functions */

//	Returns the normalizing factor for eigenvectors
template<int N>
__device__ constexpr real eigenvecNormFactor() { return const_sqrt(2.0/double(N)); }

//	Set to zero all the elements of all the strided vectors, size is the overall size of the space occupied by the vectors
template<typename T, int stride, int size>
__device__ __forceinline__ void ZeroStridedVectors(const int glob_idx, T * const d_ptr) {

	int th_id = glob_idx;

	for(;th_id<size;th_id+=stride)
		d_ptr[th_id] = (T) 0.0;
}
// Overload of previous function for float2 type
template<int stride, int size>
__device__ __forceinline__ void ZeroStridedVectors(const int glob_idx, float2 * const d_ptr) {

	int th_id = glob_idx;

	for(;th_id<size;th_id+=stride)
		d_ptr[th_id] = make_float2(0.0f,0.0f);
}
// Overload of previous function for double2 type
template<int stride, int size>
__device__ __forceinline__ void ZeroStridedVectors(const int glob_idx, double2 * const d_ptr) {

	int th_id = glob_idx;

	for(;th_id<size;th_id+=stride)
		d_ptr[th_id] = make_double2(0.0,0.0);
}

// Calculate a * b + c (where a,c are vectors of complex numbers of type T_A and b is a scalar of type T_B) and store in out
template<typename T_A, typename T_B, int stride, int size>
__device__ __forceinline__ void AddScaledCVecToCVec(const int glob_idx, T_A * const a, const T_B b, T_A * const c,T_A * const out) {

	int th_id = glob_idx;

	for(;th_id<size;th_id+=stride)
		out[th_id] = cuCadd( cuCscale<T_A,T_B>(a[th_id],b), c[th_id]);
}

// Calculate c * conj(c) of a complex number c. T_IN is the type of the complex number, T_OUT is the type of the scalar norm.
template<typename T_IN,typename T_OUT>
__device__ __forceinline__ T_OUT cuCSqrdAbs(const T_IN c) {
	return (c.x * c.x) + (c.y * c.y);
}

// Return sum of squared moduli of elements of complex vector
template<typename T_IN,typename T_OUT, int stride, int size>
__device__ __forceinline__ T_OUT CVecSqrdAbs(const int glob_idx, T_IN * const vec) {
	T_OUT sum = 0.0f;
	for(int j=glob_idx;j<size;j+=stride)
		sum += cuCSqrdAbs<cuComplex,float>(vec[j]);
	return sum;
}

// Return the norm of complex vector vec.
template<typename T_IN,typename T_OUT, int stride, int size>
__device__ __forceinline__ T_OUT CVecNorm(const int glob_idx, T_IN * const vec) {
	return sqrt(CVecSqrdAbs<T_IN,T_OUT,stride,size>(glob_idx,vec));
}

// Scale complex vector vec by factor scale and store in out
template<typename T_C, typename T_R, int stride, int size>
__device__ __forceinline__ void CVecScale(const int glob_idx, T_C * const vec,const T_R scale, T_C * const out) {
	for(int j=glob_idx;j<size;j+=stride)
		out[j] = cuCscale<T_C,T_R>(vec[j],scale);
}

// Copy vector from in to out
template<typename T, int stride, int size>
__device__ __forceinline__ void VecCopy(const int glob_idx, T * const in,T * const out) {
	for(int j=glob_idx;j<size;j+=stride)
		out[j] = in[j];
}

// Calculate the super(sub)-diagonal of H and store in out. Need time t, oscillators' amplitudes d_A, frequencies d_omega, initial phases d_phi, exponent gamma, eigenvectors' matrix d_orthU. d_osc is temporarily used to store the current oscillation of each mode.
template<int block_dim_x, int block_dim_y, int num_tot,int stride,int n, int N>
__device__ __forceinline__ void evalKappa(const int2 i, const int glob_idx, const real t, float * const d_A,real * const d_omega, float * const d_phi,const float gamma, real * const d_orthU, float * const d_osc, float * const d_out) {

	int strided_idx;
	float delta;
	for(int h=0; h<n; h++) { // for each mode
		strided_idx = stride*h+glob_idx;
		// Calculate the current time oscillation: A_h cos(omega_h t + phi_h)
		d_osc[strided_idx] = d_A[strided_idx] * __cosf(d_omega[h * num_tot + i.x] * t + d_phi[strided_idx]);
	}

	for(int j=0; j<n; j++) { //	for each site
		delta = 1.0f;
		for(int h=0; h<n; h++) { // for each mode
			strided_idx = stride*h+glob_idx;
			// Calculate the displacement by multiplying the current time oscillation times the space oscillation given by the eigenvector
			delta += d_osc[strided_idx] * d_orthU[h*n+j];
		}
		strided_idx = stride*j+glob_idx;
		// Calculate the coupling kappa_j from the displacement, with formula: (1 + delta_j)^(-gamma)
		d_out[strided_idx] = -__powf(delta, -gamma);
	}

}

//	Calculate the differential function of the schroedinger eq., output is stored in d_out. Need time t, current state d_psi, diagonal of H d_diag, oscillators' amplitudes d_A, frequencies d_omega, initial phases d_phi, exponent gamma, eigenvectors' matrix d_orthU. d_kappa is temporarily used to store the couplings, d_osc is temporarily used to store the current oscillation of each mode.
template<int block_dim_x, int block_dim_y,int num_tot,int stride,int n, int N>
__device__ __forceinline__ void diff(const int2 i, const int glob_idx, const real t, cuComplex * const d_psi,
																			float * const d_diag, float * const d_A,
																			real * const d_omega, float * const d_phi, const float gamma, real * const d_orthU,
																			float * const d_kappa, float * const d_osc, cuComplex * const d_out) {
	int c;
	int j;
	float kappa_prev;

	// Obtain sub(super)-diagonal elements of H
	evalKappa<block_dim_x,block_dim_y,num_tot,stride,n,N>(i, glob_idx,t,d_A,d_omega,d_phi,gamma,d_orthU,d_osc,d_kappa);

	// Store the product H * psi in d_out.
	// Since H is defined sparsely (by its diagonal, super and sub diagonal vectors) the product multiplies only the necessary
	// elements, ignoring the obvious zero components. Time (and space) complexity is O(N).
	c = 0;
	j = /*stride * 0 + */glob_idx; // index to element 0 of N-dimensional array
	kappa_prev = d_kappa[j];
	d_out[j] = cuCmul(diffC(),
							cuCadd(cuCscale<cuComplex,float>(d_psi[j],d_diag[(c++) * DISORDER_R + i.y%DISORDER_R]),
							cuCscale<cuComplex,float>(d_psi[j + stride], kappa_prev)) ); // First element of H*psi
	for(j += stride; j < (N-1) * stride; j+=stride) // for each internal element (2,3,...,N-1)
		d_out[j] = cuCmul(diffC(),
								cuCadd(cuCscale<cuComplex,float>(d_psi[j - stride],kappa_prev),
								cuCadd(cuCscale<cuComplex,float>(d_psi[j],d_diag[(c++) * DISORDER_R + i.y%DISORDER_R]),
								cuCscale<cuComplex,float>(d_psi[j + stride],(kappa_prev = d_kappa[j])))) ); // Internal element of H*psi
	d_out[j] = cuCmul(diffC(),
							cuCadd(cuCscale<cuComplex,float>(d_psi[j - stride],kappa_prev),
							cuCscale<cuComplex,float>(d_psi[j],d_diag[c * DISORDER_R + i.y%DISORDER_R]))); // Last element of H*psi
}

//	4th order Runge-Kutta integration method to obtain estimate for |psi(t+dt)> (stored in d_out) from |psi(t)> (d_psi). Need time t, integration step dt, current state d_psi, diagonal of H d_diag, oscillators' amplitudes d_A, frequencies d_omega, initial phases d_phi, exponent gamma, eigenvectors' matrix d_orthU. d_kappa is temporarily used to store the couplings, d_osc is temporarily used to store the current oscillation of each mode. The 4 increments are temporarily stored in d_K[4].
template<int block_dim_x, int block_dim_y,int num_tot,int stride, int n, int N>
__device__ __forceinline__ void rk4(const int2 i, const int glob_idx, const real t, const real dt,
																			cuComplex * const d_psi, float * const d_diag, float * const d_A,
																			real * const d_omega, float * const d_phi, const float gamma, real * const d_orthU,
																			float * const d_kappa, float * const d_osc, cuComplex * const d_K[4], cuComplex * const d_out) {


	int th_id = glob_idx;
	const real half_step = dt/2.0;
	const real t_half_step = t + half_step;

	/* Calculate the 4 increments (K's) of the R-K4 method */

	diff<block_dim_x,block_dim_y,num_tot,stride,n,N>(i,glob_idx,t,d_psi,d_diag,d_A,d_omega,d_phi,gamma,d_orthU,d_kappa,d_osc,d_K[0]);
	AddScaledCVecToCVec<cuComplex,real,stride,N*stride>(glob_idx,d_K[0],half_step,d_psi,d_out);
	diff<block_dim_x,block_dim_y,num_tot,stride,n,N>(i,glob_idx,t_half_step,d_out,d_diag,d_A,d_omega,d_phi,gamma,d_orthU,d_kappa,d_osc,d_K[1]);
	AddScaledCVecToCVec<cuComplex,real,stride,N*stride>(glob_idx,d_K[1],half_step,d_psi,d_out);
	diff<block_dim_x,block_dim_y,num_tot,stride,n,N>(i,glob_idx,t_half_step,d_out,d_diag,d_A,d_omega,d_phi,gamma,d_orthU,d_kappa,d_osc,d_K[2]);
	AddScaledCVecToCVec<cuComplex,real,stride,N*stride>(glob_idx,d_K[2],dt,d_psi,d_out);
	diff<block_dim_x,block_dim_y,num_tot,stride,n,N>(i,glob_idx,t+dt,d_out,d_diag,d_A,d_omega,d_phi,gamma,d_orthU,d_kappa,d_osc,d_K[3]);

	/* Do the weighted sum and save to d_out */

	for(;th_id<N*stride;th_id+=stride)
		d_out[th_id] = cuCadd(d_psi[th_id],
										cuCscale<cuComplex,real>(cuCadd(cuCadd(d_K[0][th_id], cuCscale<cuComplex,float>(cuCadd(d_K[1][th_id], d_K[2][th_id]),2.0f)),
										d_K[3][th_id]),dt/6.0));

}

// Calculate the standard deviation of the position in wavefunction d_psi.
template<int stride,int n, int N>
__device__ __forceinline__ float stateStdDev(const int glob_idx,
																										cuComplex * const d_psi) {
	float stdDev = 0.0f;
	int j;
	// Calculate the expectation value of the position.
	for(j=0;j<N;j++)
		stdDev += cuCSqrdAbs<cuComplex,float>(d_psi[glob_idx + j * stride]) * (float) (j+1);
	// Square the expectation value and take the opposite.
	stdDev = - stdDev * stdDev;
	// Sum the expectation value of the squares of the position.
	for(j=0;j<N;j++)
		stdDev += cuCSqrdAbs<cuComplex,float>(d_psi[glob_idx + j * stride]) * (float) ((j+1)*(j+1));
	// Take the square root.
	stdDev = sqrtf(stdDev);
	return stdDev;
}

// Return the probability of being in the target sites for wavefunction d_psi.
template<int stride,int n, int N>
__device__ __forceinline__ float targetProbability(const int glob_idx,
																										cuComplex * const d_psi) {

	return CVecSqrdAbs<cuComplex,float,stride,stride * (N - FIRST_TARGET_NODE)>(glob_idx,&(d_psi[stride * FIRST_TARGET_NODE]));
}


/* Device kernels */

// Find max element in array of averaged performance indexes (for each tuple (k,m,T,gamma)) and relative time index. Saves output in DATAPOINT structure out. Receives in input the grid of inputs (k_array,m_array,T_array,gamma_array) and the array of averaged performance indexes d_avg_perf_idx.
template<int num_k, int num_m, int num_T, int num_gamma, int num_tot>
__global__ void FindMaxPerfIdx(float * const k_array, real * const m_array,
																		float * const T_array, float * const gamma_array,
																		float * const d_avg_perf_idx, DATAPOINT * const out) {

	int val_idx;

	int i=blockDim.x*blockIdx.x+threadIdx.x;

	for(;i<num_tot;i+=blockDim.x * gridDim.x) { // cycle through the grid of inputs
		float max = 0.0f;
		int id_t_max = 1;
		// Find the maximum value in array d_avg_perf_idx.
		for(int j=0;j<SIM_POINTS-1;j++) {
			int data_idx = j * num_tot + i;
			float cur_perf_idx = d_avg_perf_idx[data_idx];
			if(cur_perf_idx > max) {
				max = cur_perf_idx;
				id_t_max = j+1; //d_avg_perf_idx is shifted by 1 time step with respect to other arrays like d_psi
			}
		}
		// Fill the output DATAPOINT structure.
		out[i].k = k_array[i%num_k];
		val_idx = i/num_k;
		out[i].m = m_array[val_idx%num_m];
		val_idx = val_idx/num_m;
		out[i].T = T_array[val_idx%num_T];
		val_idx = val_idx/num_T;
		out[i].gamma = gamma_array[val_idx%num_gamma];
		out[i].perf_idx = max;
		out[i].time_idx = id_t_max * SIM_DT;
	}
}

// Average performance indexes (d_perf_idx) for each (k,m,T,gamma) tuple, for 1 time point. Saves output in d_out.
// PRECONDITION: number of threads per block must be power of 2, otherwise the result will be wrong.
template<int num_tot,int threadsPerBlock>
__global__ void AveragePerfIdx(float * const d_perf_idx, float * const d_out) {
	// Using shared memory for intra-block cooperation to calculate the sum of all the elements in d_perf_idx that correspond to a tuple (k,m,T,gamma).
	__shared__ float s[threadsPerBlock]; // Dimension of array must be a power of 2

	// Each block of threads takes care of summing, among realizations, the performance indexes for 1 tuple (k,m,T,gamma).

	for(int i=blockIdx.x;i<num_tot;i+=gridDim.x) { //for each tuple (k,m,T,gamma)
		// First the sums are cumulated by each thread in a cell of shared memory.
		s[threadIdx.x] = 0.0f;
		for(int j=threadIdx.x;j<CHUNK_R;j+=blockDim.x) { // for each realization
			int data_idx = j * num_tot + i;
			s[threadIdx.x] += d_perf_idx[data_idx]; // sum current realization to cumulative (shared) variable
		}
		// Then the sums between elements of the shared memory array are carried out.
		__syncthreads();
		int count = threadsPerBlock >> 1;
		while(count > 0) {
			if(threadIdx.x < count)
				s[threadIdx.x] += s[count + threadIdx.x];
			__syncthreads();
			count >>= 1;
		}
		if(threadIdx.x == 0) {
			// The cumulated value is divided by the number of realizations and the average is saved in d_out.
			d_out[i] += s[0] / (R_CHUNKS * CHUNK_R);
		}

	}
}

//	Run CHUNK_R times num_tot parallel evolutions collecting CHUNK_R realizations of the performance index (in d_perf_idx) for each tuple (k,m,T,gamma). d_diag, d_omega, d_A, d_phi, d_orthU are expected to be already initialized. The other variables are for storing intermediate data. When requested, the deviations from norm 1 of the state are saved in d_deviations.
template<int block_dim_x, int block_dim_y, int stride_x, int stride_y, int num_k, int num_m, int num_T, int num_gamma, int num_tot,int n, int N>
__global__ void RunSim(float * const k_array, real * const m_array,
												float * const T_array, float * const gamma_array,
												cuComplex * const d_psi, float * const d_diag, float * const d_A,
												real * const d_omega, float * const d_phi, real * const d_orthU,
												float * const d_kappa, float * const d_osc, cuComplex * const d_K[4], cuComplex * const d_psi_next,
												float * const d_perf_idx /* vector of num_tot * CHUNK_R * SIM_POINTS performance indexes */,
												float * const d_deviations /* (optional) deviation from 1 of norm of est. |psi(t+dt)> */) {

	int2 i;
	float gamma;

	i.x=blockDim.x*blockIdx.x+threadIdx.x;

	for (;i.x<num_tot;i.x+=stride_x){ // for each tuple (k,m,T,gamma)

		gamma = gamma_array[(i.x/(num_k*num_m*num_T))%num_gamma]; // fetch gamma

		i.y=blockDim.y*blockIdx.y+threadIdx.y;
		for (;i.y<CHUNK_R;i.y+=stride_y){
			const int glob_idx = num_tot*i.y+i.x;

			// Init (to 0) vector of performance indexes
			ZeroStridedVectors<float,num_tot*CHUNK_R,num_tot*CHUNK_R*SIM_POINTS>(glob_idx,d_perf_idx);

			// Init |psi(0)>
			ZeroStridedVectors<num_tot * CHUNK_R,num_tot * CHUNK_R * N>(glob_idx,d_psi);
			d_psi[/* 0 * glob_stride + */ + glob_idx] = realOne();


			for(int t_id=0; t_id<SIM_POINTS; t_id++) { // for each time node
				real t = t_id * SIM_DT;
				rk4<block_dim_x,block_dim_y,num_tot,num_tot * CHUNK_R,n,N>(i,glob_idx,t,SIM_DT,d_psi,d_diag,d_A,d_omega,
						d_phi,gamma,d_orthU,d_kappa,d_osc,d_K,d_psi_next); // Calculate estimate of |psi(t+dt)>

#ifdef SIM_VERBOSE
				// Calculate deviation of norm of estimate of |psi(t+dt)> from 1.
				float norm = CVecNorm<cuComplex,float,num_tot * CHUNK_R,num_tot * CHUNK_R * N>(glob_idx,d_psi_next);
				d_deviations[t_id * (num_tot * CHUNK_R) + glob_idx] = norm - 1.0f;
#endif
				//	Estimate of |psi(t+dt)> becomes the new |psi(t)>
				VecCopy<cuComplex,num_tot * CHUNK_R,num_tot * CHUNK_R * N>(glob_idx,d_psi_next,d_psi);

				// Store performance index of |psi(t + dt)>. Later it will be averaged over realizations.
#ifdef SIM_DISORDER
				d_perf_idx[t_id * (num_tot*CHUNK_R) + glob_idx] = stateStdDev<num_tot * CHUNK_R,n,N>(glob_idx,d_psi);
#else
				d_perf_idx[t_id * (num_tot*CHUNK_R) + glob_idx] = targetProbability<num_tot * CHUNK_R,n,N>(glob_idx,d_psi);
#endif


			}
		}
	}

}

// Sample static disorder (diagonal) vectors (with variance disorder_param) for each disorder realization. Store in d_diag. Needs initialized curandState states.
template<int N>
__global__ void SampleDisorderVec(const float disorder_param, float * const d_diag, curandState * const states) {
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	int stride=blockDim.x*gridDim.x;
	for(;i<DISORDER_R;i+=stride) {
		for(int j=0;j<N;j++) {
			curandState state=states[i];
			d_diag[i * N + j] = disorder_param * curand_normal(&state);
			states[i]=state;
		}
	}
}

// Sample the amplitudes and phases for each tuple (k,m,T,gamma), mode, realization. Store the result in d_A,d_phi. Also calculate and store the mode frequencies in d_omega. Needs initialized curandState states.
template<int stride_x, int stride_y, int num_k, int num_m, int num_T, int num_gamma, int num_tot, int n, int N>
__global__ void SampleAmplitudesAndPhases(float * const k_array, real * const m_array,
														float * const T_array,
														float * const d_A, real * const d_omega, float * d_phi,
														curandState * const states) {

	int2 i;
	float k,T;
	real m;
	int val_idx;

	i.x=blockDim.x*blockIdx.x+threadIdx.x;

	for (;i.x<num_tot;i.x+=stride_x){ // loop over each (k,m,T,gamma) tuple

		// fetch values of k,m,T
		k = k_array[i.x%num_k];
		val_idx = i.x/num_k;
		m = m_array[val_idx%num_m];
		val_idx = val_idx/num_m;
		T = T_array[val_idx%num_T];

		i.y=blockDim.y*blockIdx.y+threadIdx.y;
		for (;i.y<CHUNK_R;i.y+=stride_y){ // loop over realizations
			for(int h=0;h<n;h++) { //loop over modes
				int glob_idx = num_tot*i.y+i.x;
				// Calculate frequency omega_h
				real omega = d_omega[h * num_tot + i.x] = 2.0 * sqrt(k/m) * sin(double(h+1) * PI_DOUBLE / double(2 * N) );

				real logx = -HBAR * omega / (BOLTZMANN * T);

				// Sample y,z from U[0,1]
				curandState state=states[glob_idx];
				double y = curand_uniform_double(&state);
				double z = curand_uniform_double(&state);
				states[glob_idx]=state;

				// Obtain A with inversion method (sample an exp. dist. and floor the result)
				int out_idx = h * (num_tot * CHUNK_R) + glob_idx;
				d_A[out_idx] = sqrtf( 2.0 * HBAR * omega * (floor(log(z) / logx) + 0.5) / k);

				// Sample phase
				d_phi[out_idx] = y * 2.0 * PI_DOUBLE;
			}
		}
	}
}

// Init num_gen random number generators with given seed. Output in states the generators' states.
__global__ void InitRNG(curandState *states, int seed, int num_gen){

	int i=blockDim.x*blockIdx.x+threadIdx.x;
	int stride=blockDim.x*gridDim.x;
	curandState state;

	for (;i<num_gen;i+=stride){
		state=states[i];
		curand_init(seed,i,0,&state);
		states[i]=state;
	}

}

// Calculate and store in global memory the orthogonal matrix U (size is n*n*sizeof(real), too big for constant mem.),
// columns (or rows) are the normalized eigenvectors of the coupled oscillators system. Output in d_ptr.
template<int n, int N, int n2>
__global__ void StoreOrthU(real * const d_ptr) {

	int stride=blockDim.x*gridDim.x;
	int i=blockDim.x*blockIdx.x+threadIdx.x;

	for(;i<n2;i+=stride) // for each matrix element
		d_ptr[i]=eigenvecNormFactor<N>() * sin( double((i%n + 1) * (i/n + 1)) * PI_DOUBLE / double(N) ); // calculate the matrix element
}

// Initialize a D-dimensional vector with value val for each element. Output in d_ptr.
template<typename T>
__global__  void VecInit(T * const d_ptr, const T val, const int D){

	int stride=blockDim.x*gridDim.x;
	int i=blockDim.x*blockIdx.x+threadIdx.x;

	for (;i<D;i+=stride)
		d_ptr[i]=val;
}


int main(int argc, char **argv){

	/* Declare and initialize variables. */

	OutputCollector output;
	NormErrorCollector normError;

	float const k_array[] = PARAM_K_VALUES;
	real const m_array[] = PARAM_M_VALUES;
	float const T_array[] = PARAM_T_VALUES;
	float const gamma_array[] = PARAM_GAMMA_VALUES;

	const int seed = RAND_SEED;
	const int N = N_SITES;
	const int n = N_MODES;
	const int n2 = N_MODES2;
	const int RNGthreadsPerBlock = 1024;
	const int RNGnum_blocks = 256;
	const int AVGthreadsPerBlock = 1024;
	const int AVGnum_blocks = 64;
	const int FNDMAXthreadsPerBlock = 1024;
	const int FNDMAXnum_blocks = 256;
	const int num_k = sizeof_array(k_array);
	const int num_m = sizeof_array(m_array);
	const int num_T = sizeof_array(T_array);
	const int num_gamma = sizeof_array(gamma_array);
	const int chunk_num_k = CHUNK_SIZE_K;
	const int chunk_num_m = CHUNK_SIZE_M;
	const int chunk_num_T = CHUNK_SIZE_T;
	const int chunk_num_gamma = CHUNK_SIZE_GAMMA;
	const int chunk_size = chunk_num_k * chunk_num_m * chunk_num_T * chunk_num_gamma;
	const int threadsPerBlockX = 32, threadsPerBlockY = 8;
	const int blocksPerGridX = 2, blocksPerGridY = 60;
	const int stride_x = threadsPerBlockX * blocksPerGridX;
	const int stride_y = threadsPerBlockY * blocksPerGridY;

	const dim3 threadsPerBlock2D(threadsPerBlockX,threadsPerBlockY);
	const dim3 blocksPerGrid2D(blocksPerGridX,blocksPerGridY);

	handleCudaError(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)); // Set cache configuration to 48KB of L1 and 16KB of shared mem

	cuComplex * K_arr[4];
	DATAPOINT * datapoints = (DATAPOINT *)malloc(chunk_size * sizeof(DATAPOINT));
	float * deviations = (float *) malloc(chunk_size*CHUNK_R*SIM_POINTS * sizeof(float));

	cuComplex *d_psi, *d_psi_next;
	cuComplex **d_K;
	float *d_k_array,*d_T_array,*d_gamma_array,*d_diag,*d_A,*d_phi,*d_kappa,*d_perf_idx,*d_deviations,*d_avg_perf_idx,*d_osc;
	real *d_m_array, *d_omega, *d_orthU;
	curandState *d_states;
	DATAPOINT * d_datapoints;
	size_t free_mem,total_mem;
	// Display some information about free device memory before allocating variables
	handleCudaError(cudaMemGetInfo(&free_mem, &total_mem));
	printf("Before Malloc: occupied mem is %.2f MB\n", float(total_mem - free_mem) / 1000000.0f);
	fflush(stdout);
	// Allocate arrays on device memory
	handleCudaError(cudaMalloc((void **)&d_orthU, n2 * sizeof(real)));
	handleCudaError(cudaMalloc((void **)&d_k_array,chunk_num_k * sizeof(float)));
	handleCudaError(cudaMalloc((void **)&d_m_array,chunk_num_m * sizeof(real)));
	handleCudaError(cudaMalloc((void **)&d_T_array,chunk_num_T * sizeof(float)));
	handleCudaError(cudaMalloc((void **)&d_gamma_array,chunk_num_gamma * sizeof(float)));
	handleCudaError(cudaMalloc((void **)&d_psi, chunk_size * CHUNK_R * N * sizeof(cuComplex)));
	handleCudaError(cudaMalloc((void **)&d_diag, N * DISORDER_R * sizeof(float)));
	handleCudaError(cudaMalloc((void **)&d_omega, n * chunk_size * sizeof(real)));
	handleCudaError(cudaMalloc((void **)&d_A, CHUNK_R * n * chunk_size * sizeof(float)));
	handleCudaError(cudaMalloc((void **)&d_phi, CHUNK_R * n * chunk_size * sizeof(float)));
	handleCudaError(cudaMalloc((void **)&d_osc, CHUNK_R * n * chunk_size * sizeof(float)));
	handleCudaError(cudaMalloc((void **)&d_states, CHUNK_R * chunk_size * sizeof(curandState)));
	handleCudaError(cudaMalloc((void **)&d_kappa, n * chunk_size * CHUNK_R * sizeof(float)));
	handleCudaError(cudaMalloc((void **)&d_K, sizeof(cuComplex *) * 4));
	for(int i=0;i<4;i++)
		handleCudaError(cudaMalloc((void **)&(K_arr[i]), chunk_size * CHUNK_R * N * sizeof(cuComplex)));
	handleCudaError(cudaMalloc((void **)&d_psi_next, chunk_size * CHUNK_R * N * sizeof(cuComplex)));
	handleCudaError(cudaMalloc((void **)&d_perf_idx, chunk_size*CHUNK_R*SIM_POINTS * sizeof(float)));
	#ifdef SIM_VERBOSE
		handleCudaError(cudaMalloc((void **)&d_deviations, chunk_size*CHUNK_R*SIM_POINTS * sizeof(float)));
	#else
		d_deviations = NULL;
	#endif
	handleCudaError(cudaMalloc((void **)&d_avg_perf_idx, chunk_size*SIM_POINTS * sizeof(float)));
	handleCudaError(cudaMalloc((void **)&d_datapoints, chunk_size * sizeof(DATAPOINT)));
	CUDASYNC();
	// Display some information about free device memory after allocating variables
	handleCudaError(cudaMemGetInfo(&free_mem, &total_mem));
	printf("After Malloc: occupied mem is %.2f MB\n", float(total_mem - free_mem) / 1000000.0f);
	fflush(stdout);
	printf("CUDA memory allocation completed\n");
	fflush(stdout);

	// Split the grid of inputs in chunks.
	const int k_chunks = (num_k + chunk_num_k - 1) / chunk_num_k;
	const int m_chunks = (num_m + chunk_num_m - 1) / chunk_num_m;
	const int T_chunks = (num_T + chunk_num_T - 1) / chunk_num_T;
	const int gamma_chunks = (num_gamma + chunk_num_gamma - 1) / chunk_num_gamma;
	const int tot_chunks = k_chunks * m_chunks * T_chunks * gamma_chunks;

	CUDASYNC();
	// Calculate orthogonal matrix U of normalized eigenvectors of the coupled oscillators.
	StoreOrthU<n,N,n2><<<n,n>>>(d_orthU);
	// Initialize the random number generators.
	InitRNG<<<RNGnum_blocks,RNGthreadsPerBlock>>>(d_states,seed,CHUNK_R * chunk_size);
	CUDASYNC();
#ifdef SIM_DISORDER
	// If the diagonal elements of the Hamiltonian need to be disordered, sample their values,
	SampleDisorderVec<N><<<1,ARCH_MAX_THREADS>>>(DISORDER_PARAM, d_diag, d_states);
#else
	// otherwise set them to 0.
	cudaMemset(d_diag,0, N * DISORDER_R * sizeof(float));
#endif

	float pb_progress = 0.0f;
	const int pb_barWidth = 50;

	// Start performance timer
	std::chrono::high_resolution_clock::time_point perf_start_time = std::chrono::high_resolution_clock::now();

	for(int chunk_id = 0; chunk_id < tot_chunks; chunk_id++) { // cycle through chunks of inputs
		// Fetch chunk.
		int chunk_k_idx = chunk_id%k_chunks;
		float const * chunk_k_array = &k_array[chunk_k_idx * chunk_num_k];
		int val_idx = chunk_id/k_chunks;
		int chunk_m_idx = val_idx%m_chunks;
		real const * chunk_m_array = &m_array[chunk_m_idx * chunk_num_m];
		val_idx = val_idx/m_chunks;
		int chunk_T_idx = val_idx%T_chunks;
		float const * chunk_T_array = &T_array[chunk_T_idx * chunk_num_T];
		val_idx = val_idx/T_chunks;
		int chunk_gamma_idx = val_idx%gamma_chunks;
		float const * chunk_gamma_array = &gamma_array[chunk_gamma_idx * chunk_num_gamma];
		// Calculate effective number of elements in chunk.
		int eff_chunk_num_k = min((chunk_k_idx+1) * chunk_num_k, num_k) - chunk_k_idx * chunk_num_k;
		int eff_chunk_num_m = min((chunk_m_idx+1) * chunk_num_m, num_m) - chunk_m_idx * chunk_num_m;
		int eff_chunk_num_T = min((chunk_T_idx+1) * chunk_num_T, num_T) - chunk_T_idx * chunk_num_T;
		int eff_chunk_num_gamma = min((chunk_gamma_idx+1) * chunk_num_gamma, num_gamma) - chunk_gamma_idx * chunk_num_gamma;

		// Notify the (eventual) profiler to start profiling now.
		cudaProfilerStart();
		// Initialize iteration variables.
		handleCudaError(cudaMemset(d_k_array,1,chunk_num_k * sizeof(float)));
		handleCudaError(cudaMemset(d_m_array,1,chunk_num_m * sizeof(real)));
		handleCudaError(cudaMemset(d_T_array,1,chunk_num_T * sizeof(float)));
		handleCudaError(cudaMemset(d_gamma_array,1,chunk_num_gamma * sizeof(float)));
		handleCudaError(cudaMemcpy(d_k_array,chunk_k_array,eff_chunk_num_k * sizeof(float),cudaMemcpyHostToDevice));
		handleCudaError(cudaMemcpy(d_m_array,chunk_m_array,eff_chunk_num_m * sizeof(real),cudaMemcpyHostToDevice));
		handleCudaError(cudaMemcpy(d_T_array,chunk_T_array,eff_chunk_num_T* sizeof(float),cudaMemcpyHostToDevice));
		handleCudaError(cudaMemcpy(d_gamma_array,chunk_gamma_array,eff_chunk_num_gamma * sizeof(float),cudaMemcpyHostToDevice));
		handleCudaError(cudaMemcpy(d_K, K_arr, sizeof(cuComplex *) * 4, cudaMemcpyHostToDevice));
		CUDASYNC();
		// Calculate the frequencies and sample amplitudes and phases for each tuple of inputs in the chunk, mode, realization.
		SampleAmplitudesAndPhases<stride_x, stride_y, chunk_num_k, chunk_num_m, chunk_num_T, chunk_num_gamma, chunk_size, n, N><<<blocksPerGrid2D,threadsPerBlock2D>>>(d_k_array,d_m_array,d_T_array,
																d_A, d_omega, d_phi, d_states);
		CUDASYNC();
		// Set cache configuration to 48KB of L1 and 16KB of shared mem for kernel RunSim<...>
		handleCudaError(cudaFuncSetCacheConfig(RunSim<threadsPerBlockX,threadsPerBlockY,stride_x,stride_y,chunk_num_k,chunk_num_m,chunk_num_T,chunk_num_gamma,chunk_size,n,N>,cudaFuncCachePreferL1));
		cudaMemset(d_avg_perf_idx,0,chunk_size*SIM_POINTS * sizeof(float));

		for(int r_chunk_id = 0; r_chunk_id < R_CHUNKS; r_chunk_id++) { // cycle through chunks of realizations
			// Run parallel evolutions for each tuple in the input chunk and for each realization in the realization chunk. Obtain per-evolution performance indexes in d_perf_idx.
			RunSim<threadsPerBlockX,threadsPerBlockY,stride_x,stride_y,chunk_num_k,chunk_num_m,chunk_num_T,chunk_num_gamma,chunk_size,n,N><<<blocksPerGrid2D,threadsPerBlock2D>>>(d_k_array, d_m_array,d_T_array, d_gamma_array,
															 d_psi,  d_diag,  d_A, d_omega,  d_phi,  d_orthU, d_kappa, d_osc, d_K,  d_psi_next,
															 d_perf_idx, d_deviations);
			CUDASYNC();
			#ifdef SIM_VERBOSE
					handleCudaError(cudaMemcpy(deviations, d_deviations, chunk_size*CHUNK_R*SIM_POINTS * sizeof(float), cudaMemcpyDeviceToHost));
					normError.add_values<chunk_size*CHUNK_R*SIM_POINTS>(deviations);
			#endif
			// Set cache configuration to 16KB of L1 and 48KB of shared mem for kernel AveragePerfIdx<...>
			handleCudaError(cudaFuncSetCacheConfig(AveragePerfIdx<chunk_size,AVGthreadsPerBlock>, cudaFuncCachePreferShared));
			for(int i=0;i<SIM_POINTS;i++) // Cycle through time points. This kernel's instances may run in parallel
				// Average the performance indexes obtained among realizations for current time point.
				AveragePerfIdx<chunk_size,AVGthreadsPerBlock><<<AVGnum_blocks,AVGthreadsPerBlock>>>(&(d_perf_idx[chunk_size*CHUNK_R*i]),&(d_avg_perf_idx[chunk_size*i]));
			CUDASYNC();

		}
		// Find the max with respect to time of the performance index and relative time index of max.
		FindMaxPerfIdx<chunk_num_k,chunk_num_m,chunk_num_T,chunk_num_gamma,chunk_size><<<FNDMAXnum_blocks,FNDMAXthreadsPerBlock>>>(d_k_array,d_m_array,
																																						d_T_array,d_gamma_array,
																																						d_avg_perf_idx,d_datapoints);
		CUDASYNC();
		//	Copy result back to host.
		handleCudaError(cudaMemcpy(datapoints, d_datapoints, chunk_size * sizeof(DATAPOINT), cudaMemcpyDeviceToHost));

		CUDASYNC();
		//	Tell the (eventual) profiler to stop profiling now.
		cudaProfilerStop();
		//	Collect the outputs in a data structure.
		for(int dp_id=0;dp_id<chunk_size;dp_id++) {
			const int k_idx = dp_id%chunk_num_k;
			val_idx = dp_id/chunk_num_k;
			const int m_idx = val_idx%chunk_num_m;
			val_idx = val_idx/chunk_num_m;
			const int T_idx = val_idx%chunk_num_T;
			val_idx = val_idx/chunk_num_T;
			const int gamma_idx = val_idx%chunk_num_gamma;
			if(k_idx >= eff_chunk_num_k || m_idx >= eff_chunk_num_m || T_idx >= eff_chunk_num_T || gamma_idx >= eff_chunk_num_gamma) continue;
			DATAPOINT cur = datapoints[dp_id];

			output.add_datapoint(cur);
		}


		// Check performance timer
		std::chrono::high_resolution_clock::time_point perf_stop_time = std::chrono::high_resolution_clock::now();
		float perf_duration = std::chrono::duration_cast<std::chrono::seconds>( perf_stop_time - perf_start_time ).count();
		int eta_seconds = perf_duration * (1.0f/(float(chunk_id + 1) / float(tot_chunks)) - 1.0f);

		pb_progress = float(chunk_id + 1) / float(tot_chunks);

		std::cout << "[";
    int pos = pb_barWidth * pb_progress;
    for (int i = 0; i < pb_barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setw( 0 ) << std::setprecision( 2 ) << \
		std::setfill( '0' ) << float(pb_progress * 100.0) << 	"% " << \
		"ETA: "<< std::setw( 2 ) << eta_seconds/3600 << ":" << (eta_seconds%3600)/60 << ":" << (eta_seconds%3600)%60 << "\t\r";
    std::cout.flush();

	}
	std::cout << "\n";
	std::cout.flush();
	/* Free the device and host memories. */
	handleCudaError(cudaFree((void *)d_orthU));
	handleCudaError(cudaFree((void *)d_k_array));
	handleCudaError(cudaFree((void *)d_m_array));
	handleCudaError(cudaFree((void *)d_T_array));
	handleCudaError(cudaFree((void *)d_gamma_array));
	handleCudaError(cudaFree((void *)d_psi));
	handleCudaError(cudaFree((void *)d_diag));
	handleCudaError(cudaFree((void *)d_omega));
	handleCudaError(cudaFree((void *)d_A));
	handleCudaError(cudaFree((void *)d_phi));
	handleCudaError(cudaFree((void *)d_osc));
	handleCudaError(cudaFree((void *)d_states));
	handleCudaError(cudaFree((void *)d_kappa));
	handleCudaError(cudaFree((void *)d_K));
	for(int i=0;i<4;i++)
		handleCudaError(cudaFree((void *)K_arr[i]));
	handleCudaError(cudaFree((void *)d_psi_next));
	handleCudaError(cudaFree((void *)d_perf_idx));
	#ifdef SIM_VERBOSE
		handleCudaError(cudaFree((void *)d_deviations));
	#endif
	handleCudaError(cudaFree((void *)d_avg_perf_idx));
	handleCudaError(cudaFree((void *)d_datapoints));
	free(datapoints);

	CUDASYNC();

	printf("Simulation complete. Saving to file.\n");
	fflush(stdout);

	// Save to file the results.

	cudaDeviceReset();
#ifdef SIM_VERBOSE
		if(argc==2)
			normError.saveCsv(argv[1] + std::string("_norm_err.csv"));
		else
			normError.saveCsv(std::string("norm_err.csv"));
#endif
	if(argc==2)
		output.saveCsvs(argv[1] + std::string("_perf_idx.csv"),argv[1] + std::string("_time_idx.csv"));
	 else
		output.saveCsvs(std::string("perf_idx.csv"),std::string("time_idx.csv"));

}
