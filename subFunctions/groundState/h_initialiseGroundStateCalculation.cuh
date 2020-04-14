__global__
void d_prepNormXT (
	wfStruct psi,
	tempStruct temp
)
{
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	int j = (blockDim.y * blockIdx.y) + threadIdx.y;
  	int idx = i + j * N_x;

	temp.d_D1[idx] = (pow(psi.d_x[idx].x, 2.0) + pow(psi.d_x[idx].y, 2.0)) * dx * dx;
}

__global__
void d_prepNormX (
	wfStruct psi,
	tempStruct temp
)
{
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	int j = (blockDim.y * blockIdx.y) + threadIdx.y;
  	int idx = i + j * N_x;

	temp.d_D1[idx] = (pow(psi.d_x[idx].x, 2.0)) * dx * dx;
}

__global__
void d_prepNormK (
	wfStruct psi,
	tempStruct temp
)
{
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	int j = (blockDim.y * blockIdx.y) + threadIdx.y;
  	int idx = i + j * N_x;

	temp.d_D1[idx] = (pow(psi.d_k[idx].x, 2.0) + pow(psi.d_k[idx].y, 2.0)) * dk * dk;
}

__global__
void d_setImaginaryToZero (
	double2 * f
)
{
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	int j = (blockDim.y * blockIdx.y) + threadIdx.y;
  	int idx = i + j * N_x;

	f[idx].y = 0.0;
}

double normalizeWaveFunction (
	wfStruct psi,
	paramsStruct params,
	tempStruct temp,
	handleStruct handles
)
{
	dim3 blocks (N_x / MTPB, N_x / MTPB, 1);
	dim3 thread (MTPB, MTPB, 1);

	double norm, energy;

	d_prepNormX <<< blocks, thread >>> (psi, temp);

	cublasDasum(handles.h_blas, N_x * N_x, temp.d_D1, 1, & norm);

	energy = log(norm) / dt / 2.0;
	norm = 1.0 / sqrt(norm) / N_x / N_x;

	cublasZdscal(handles.h_blas, N_x * N_x, & norm, psi.d_x, 1);

	return - energy * 27.212;
}

void h_getNorm (
	wfStruct psi,
	paramsStruct params,
	tempStruct temp,
	handleStruct handles,
	int tdx
)
{
	dim3 blocks (N_x / MTPB, N_x / MTPB, 1);
	dim3 thread (MTPB, MTPB, 1);

	d_prepNormXT <<< blocks, thread >>> (psi, temp);

	cublasDasum(handles.h_blas, N_x * N_x, temp.d_D1, 1, & psi.h_norm[tdx]);
}

__global__
void multiplyElementsIR(double2 * f, double * g)
{
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	int j = (blockDim.y * blockIdx.y) + threadIdx.y;
  	int index = i + j * N_x;

	f[index].x *= g[index];
  	f[index].y *= 0.0;
}

__global__
void fftShift2D(double2 * f)
{
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	int j = (blockDim.y * blockIdx.y) + threadIdx.y;
  	int index = i + j * N_x;

  if (i < N_x && j < N_x)
    {
       double a = 1 - 2 * ((i + j) & 1);
       f[index].x *= a;
       f[index].y *= a;
   }
}

void h_calculateGroundState (
	wfStruct psi,
	hamiltonianStruct hamiltonian,
	paramsStruct params,
	tempStruct temp,
	handleStruct handles
)
{
	dim3 blocks (N_x / MTPB, N_x / MTPB, 1);
	dim3 thread (MTPB, MTPB, 1);

	double energy, energy_new, convergence;

    cufftHandle plan;
    cufftPlan2d(& plan, N_x, N_x, CUFFT_Z2Z);

	convergence = 1E6;

	while (convergence > convergenceFactor)
	{
		printf("\r  Ground State Convergence: %.2e \n", convergence);
	    printf ("\033[1A");
	    printf ("\033[K");

		convergence = convergence * 0.9999;

		multiplyElementsIR <<< blocks, thread >>> (psi.d_x, hamiltonian.d_iV);
		cudaThreadSynchronize();

	    fftShift2D <<< blocks, thread >>> (psi.d_x);
	    cudaThreadSynchronize();

		cufftExecZ2Z(plan, psi.d_x, psi.d_k, CUFFT_FORWARD);
	    cudaThreadSynchronize();

		d_setImaginaryToZero <<< blocks, thread >>> (psi.d_k);

		multiplyElementsIR <<< blocks, thread >>> (psi.d_k, hamiltonian.d_iT);
	    cudaThreadSynchronize();

	    cufftExecZ2Z(plan, psi.d_k, psi.d_x, CUFFT_INVERSE);
	    cudaThreadSynchronize();

		d_setImaginaryToZero <<< blocks, thread >>> (psi.d_x);

	    fftShift2D <<< blocks, thread >>> (psi.d_k);
	    cudaThreadSynchronize();

	    fftShift2D <<< blocks, thread >>> (psi.d_x);
	    cudaThreadSynchronize();

	    multiplyElementsIR <<< blocks, thread >>> (psi.d_x, hamiltonian.d_iV);
	    cudaThreadSynchronize();

		energy_new = normalizeWaveFunction (psi, params, temp, handles);
		convergence = fabs(energy - energy_new);
		energy = energy_new;
	}

	printf("\n     Ionization Potential Energy: %.2lf eV\n", energy);
}

void h_initialiseGroundStateCalculation(
	wfStruct psi,
	hamiltonianStruct hamiltonian,
	paramsStruct params,
	tempStruct temp,
	handleStruct handles
)
{
	printf("  3. Ground State Calculation \n");
	h_calculateGroundState (psi, hamiltonian, params, temp, handles);
}
