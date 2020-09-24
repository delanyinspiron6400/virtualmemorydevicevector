#include <iostream>
#include <vector>

#include "VirtualMemoryVector.cuh"

template <typename T>
__global__ void d_testwrite(T* __restrict data, unsigned int num_elements)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_elements)
		return;

	data[tid] = tid;
}

template <typename T>
__global__ void d_testread(const T* __restrict data, unsigned int num_elements)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_elements)
		return;

	auto element = data[tid];
	if(tid == 0)
		printf("Read worked with TID: %u\n", element);
}


int main()
{
	int device{0};
	VirtMem virtual_memory;
	virtual_memory.initialize(device);

	// Lets allocate 1 GiB
	size_t size = 1ULL * 1024ULL * 1024ULL * 1024ULL;
	size_t address_size = 8ULL * 1024ULL * 1024ULL * 1024ULL;

	virtual_memory.reserveSize(address_size);
	unsigned int* d_ptr = reinterpret_cast<unsigned int*>(virtual_memory.allocMem(size));

	// Lets try to write to memory now
	unsigned int num_elements{static_cast<unsigned int>(virtual_memory.alloc_size / sizeof(unsigned int))};
	unsigned int blockSize{256};
	unsigned int gridSize{cuHelper::divup(num_elements, blockSize)};

	d_testwrite<<<gridSize, blockSize>>>(d_ptr, num_elements);
	HANDLE_ERROR(cudaDeviceSynchronize());
	d_testread<<<gridSize, blockSize>>>(d_ptr, num_elements);
	HANDLE_ERROR(cudaDeviceSynchronize());

	virtual_memory.allocMem(size);

	num_elements = static_cast<unsigned int>(virtual_memory.alloc_size / sizeof(unsigned int));

	d_testwrite<<<gridSize, blockSize>>>(d_ptr, num_elements);
	HANDLE_ERROR(cudaDeviceSynchronize());
	d_testread<<<gridSize, blockSize>>>(d_ptr, num_elements);
	HANDLE_ERROR(cudaDeviceSynchronize());

	unsigned int* test_ptr{nullptr};
	size_t address_size_runtime = 6ULL * 1024ULL * 1024ULL * 1024ULL;
	HANDLE_ERROR(cudaMalloc(&test_ptr, address_size_runtime));
	num_elements = static_cast<unsigned int>(address_size_runtime / sizeof(unsigned int));
	d_testwrite<<<gridSize, blockSize>>>(test_ptr, num_elements);
	HANDLE_ERROR(cudaDeviceSynchronize());
	d_testread<<<gridSize, blockSize>>>(test_ptr, num_elements);
	HANDLE_ERROR(cudaDeviceSynchronize());


	return 0;
}