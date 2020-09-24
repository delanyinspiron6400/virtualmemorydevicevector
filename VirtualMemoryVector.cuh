#pragma once

#include "Helper.cuh"

struct VMAllocs
{
	CUdeviceptr ptr;
		size_t size;
};

struct VirtMem
{
	size_t granularity{0ULL};
	std::vector<CUmemGenericAllocationHandle> handles;
	std::vector<VMAllocs> allocs;
	CUmemAllocationProp prop{};
	CUmemAccessDesc accessDesc{};
	CUdeviceptr ptr{0ULL};
	size_t alloc_size{0ULL};
	size_t reserve_size{0ULL};

	~VirtMem()
	{
		for(const auto& alloc : allocs)
			HANDLE_ERROR_DAPI(cuMemUnmap(alloc.ptr, alloc.size));
		HANDLE_ERROR_DAPI(cuMemAddressFree(ptr, reserve_size));
		for(const auto handle : handles)
			HANDLE_ERROR_DAPI(cuMemRelease(handle));
			
	}

	void initialize(int device)
	{
		prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id = device;
		accessDesc.location = prop.location;
		accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

		HANDLE_ERROR(cudaSetDevice(device));

		int deviceSupportsVmm;
		HANDLE_ERROR_DAPI(cuDeviceGetAttribute(&deviceSupportsVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device));
		if(deviceSupportsVmm)
			std::cout << "Virtual Memory Management supported!" << std::endl;
		else
		{
			std::cout << "Virtual Memory Management NOT supported!" << std::endl;
			return;
		}

		// Compute the granularity that the device offers
		HANDLE_ERROR_DAPI(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
	}

	void reserveSize(size_t size)
	{
		reserve_size = cuHelper::align(size, granularity);
		HANDLE_ERROR_DAPI(cuMemAddressReserve(&ptr, reserve_size, 0, 0, 0));
	}

	void tryIncreaseReserve(size_t additional_size)
	{
		CUdeviceptr new_ptr;
		additional_size = cuHelper::align(additional_size, granularity);
		// Try first to get some addresses at the back of the current addresses
		auto status = cuMemAddressReserve(&new_ptr, additional_size, 0ULL, ptr + reserve_size, 0ULL);
		if(status != CUDA_SUCCESS || (new_ptr != (ptr + reserve_size)))
		{
			// Increase did not work, check if we got a working pointer, which we now do not need
			if(new_ptr != 0ULL)
			{
				HANDLE_ERROR_DAPI(cuMemAddressFree(new_ptr, additional_size));
			}

			size_t new_size = additional_size + reserve_size;
			status = cuMemAddressReserve(&new_ptr, new_size, 0ULL, 0, 0);
			if(status == CUDA_SUCCESS && ptr != 0ULL)
			{
				if(ptr != 0ULL)
				{
					// TODO!
					std::cout << "Reallocation path not implemented fully yet!\n";
					exit(-1);
				}

				ptr = new_ptr;
				reserve_size = new_size;
				allocs.push_back(VMAllocs{new_ptr, new_size});
			}
		}
		else 
		{
			allocs.push_back(VMAllocs{new_ptr, additional_size});
			if (ptr == 0ULL) {
				ptr = new_ptr;
			}
			reserve_size = additional_size + reserve_size;
		}
	}

	void* allocMem(size_t size)
	{
		auto padded_size = cuHelper::align(size, granularity);
		if(padded_size + alloc_size > reserve_size)
		{
			std::cout << "Increasing the address range does not currently work yet!" << std::endl;
			tryIncreaseReserve((padded_size + alloc_size) - reserve_size);
			return nullptr;
		}
		CUmemGenericAllocationHandle allocHandle;
		HANDLE_ERROR_DAPI(cuMemCreate(&allocHandle, padded_size, &prop, 0));
		handles.push_back(allocHandle);
		HANDLE_ERROR_DAPI(cuMemMap(ptr + alloc_size, padded_size, 0, allocHandle, 0));
		allocs.push_back(VMAllocs{ptr + alloc_size, padded_size});
		void* ret_ptr{reinterpret_cast<void*>(ptr + alloc_size)};
		HANDLE_ERROR_DAPI(cuMemSetAccess(ptr + alloc_size, padded_size, &accessDesc, 1));
		alloc_size += padded_size;
		return ret_ptr;
	}
};