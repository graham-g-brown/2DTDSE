void h_checkCUDADevice() {
    int devID = CUDA_DEVICE;
    int deviceCount;
    cudaDeviceProp deviceProp;
	int cudaVersion;

    cudaGetDeviceCount(&deviceCount);
    cudaGetDeviceProperties(&deviceProp, devID);

	size_t memFree, total;

    if (!cudaSetDevice(devID))
    {
		cudaMemGetInfo(& memFree,& total);
		cudaDriverGetVersion ( & cudaVersion);

		printf("\n");
		printf("  2. Computation Platform Information \n\n");
		printf("     CUDA Version       : %d.%d \n", cudaVersion / 1000, (cudaVersion % 100) / 10);
		printf("     Platform           : NVIDIA-CUDA\n");
        printf("     Device %d (arch %d%d) : %s\n",
               devID, deviceProp.major, deviceProp.minor,  deviceProp.name);
		printf("     VRAM               : %05lu MB \n", deviceProp.totalGlobalMem/1024/1024);
		printf("     Multiprocessors    : %d \n", deviceProp.multiProcessorCount);
		printf("     Shared RAM / Block : %lu \n",deviceProp.sharedMemPerBlock);
		printf("     Registers / Block  : %d \n", deviceProp.regsPerBlock);
		printf("     Warp size          : %d\n", deviceProp.warpSize);
		printf("\n");
    }

    else
    {
		printf("\n");
        printf("  Failure setting CUDA Device\n");
        printf("\n");
        exit(0);
    }
}
