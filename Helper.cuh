#pragma once

#include <cuda.h>

namespace cuHelper
{
	// ##############################################################################################################################################
	//
	static inline void HandleError(cudaError_t err,
		const char* string,
		const char *file,
		int line) {
		if (err != cudaSuccess) {
			printf("%s\n", string);
			printf("%s in \n%s at line %d\n", cudaGetErrorString(err),
				file, line);
			exit(EXIT_FAILURE);
		}
	}

	// ##############################################################################################################################################
	//
	static inline void HandleError_driver_api(CUresult err,
		const char* string,
		const char *file,
		int line) {
		if (err != CUDA_SUCCESS) {
			printf("%s\n", string);
			const char* errorstring;
			cuGetErrorString(err, &errorstring);
			printf("%s in \n%s at line %d\n", errorstring,
				file, line);
			exit(EXIT_FAILURE);
		}
	}

	template<typename T>
	__host__ __device__ constexpr T divup(T a, T b)
	{
		return (a + b - 1) / b;
	}

	template<typename T>
	__host__ __device__ constexpr T align(T a, T b)
	{
		return divup(a, b) * b;
	}
}

#define HANDLE_ERROR( err ) (cuHelper::HandleError( err, "", __FILE__, __LINE__ ))
#define HANDLE_ERROR_DAPI( err ) (cuHelper::HandleError_driver_api( err, "", __FILE__, __LINE__ ))