#include <limits>

// Calculate at compile time the size of a (compile-time) initialized array.
template <typename T>
constexpr auto sizeof_array(const T& iarray) -> std::size_t {
    return (sizeof(iarray) / sizeof(iarray[0]));
}

/* Compile-time double square root */

namespace Detail
{
	double constexpr sqrtNewtonRaphson(double x, double curr, double prev)
	{
		return curr == prev
			? curr
			: sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
	}
}

/*
* Constexpr version of the square root
* Return value:
*	- For a finite and non-negative value of "x", returns an approximation for the square root of "x"
*   - Otherwise, returns NaN
*/
double constexpr const_sqrt(double x)
{
	return x >= 0 && x < std::numeric_limits<double>::infinity()
		? Detail::sqrtNewtonRaphson(x, x, 0)
		: std::numeric_limits<double>::quiet_NaN();
}

/* cuda error handling function */
#define handleCudaError(ans) { cudaErrorHandler((ans), __FILE__, __LINE__); }
void cudaErrorHandler(cudaError_t cudaERR, const char *file, int line, bool abort=true){
  if (cudaERR!=cudaSuccess){
    fprintf(stdout,"CUDA error: %s %s %d\n", cudaGetErrorString(cudaERR), file, line);
		fflush(stdout);
		fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(cudaERR), file, line);
		fflush(stderr);
		if (abort) exit(cudaERR);
  }
}
