__global__ void multiplyElementsInt
(
	wfStruct psi,
	paramsStruct params,
	int tdx
)
{

  double a, b, v, v_r, v_i;

  int i = (blockDim.x * blockIdx.x) + threadIdx.x;
  int j = (blockDim.y * blockIdx.y) + threadIdx.y;
  int idx = i + j * N_x;

  v   = - params.d_E[tdx] * (params.d_x[i] + params.d_x[j]) * dt;
  v_r =   cos(v);
  v_i = - sin(v);

  a = psi.d_x[idx].x * v_r - psi.d_x[idx].y * v_i;
  b = psi.d_x[idx].x * v_i + psi.d_x[idx].y * v_r;

  psi.d_x[idx].x = a * exp(- params.d_abc[i] - params.d_abc[j]);
  psi.d_x[idx].y = b * exp(- params.d_abc[i] - params.d_abc[j]);
}

__global__
void multiplyElementsHPsiPHalfStep
(
	wfStruct psi,
	hamiltonianStruct hamiltonian,
	paramsStruct params,
	int tdx
)
{

  double a, b, v, v_r, v_i;

  int i = (blockDim.x * blockIdx.x) + threadIdx.x;
  int j = (blockDim.y * blockIdx.y) + threadIdx.y;
  int idx = i + j * N_x;

  v   = hamiltonian.d_V[idx].x - 0.5 * params.d_E[tdx] * (params.d_x[i] + params.d_x[j]) * dt;
  v_r = cos(v);
  v_i = - sin(v);

  a = psi.d_x[idx].x * v_r - psi.d_x[idx].y * v_i;
  b = psi.d_x[idx].x * v_i + psi.d_x[idx].y * v_r;

  psi.d_x[idx].x = a * exp(- params.d_abc[i] - params.d_abc[j]);
  psi.d_x[idx].y = b * exp(- params.d_abc[i] - params.d_abc[j]);
}

__global__
void multiplyElementsHPsiP
(
	wfStruct psi,
	hamiltonianStruct hamiltonian,
	paramsStruct params,
	int tdx
)
{

  double a, b, v, v_r, v_i;

  int i = (blockDim.x * blockIdx.x) + threadIdx.x;
  int j = (blockDim.y * blockIdx.y) + threadIdx.y;
  int idx = i + j * N_x;

  v   = 2.0 * (hamiltonian.d_V[idx].x - 0.5 * params.d_E[tdx] * (params.d_x[i] + params.d_x[j]) * dt);
  v_r = cos(v);
  v_i = - sin(v);

  a = psi.d_x[idx].x * v_r - psi.d_x[idx].y * v_i;
  b = psi.d_x[idx].x * v_i + psi.d_x[idx].y * v_r;

  psi.d_x[idx].x = a * exp(- params.d_abc[i] - params.d_abc[j]);
  psi.d_x[idx].y = b * exp(- params.d_abc[i] - params.d_abc[j]);
}

__global__ void divideElements(double2 * f, double g)
{
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    int j = (blockDim.y * blockIdx.y) + threadIdx.y;
    int idx = i + j * N_x;
  	f[idx].x /= g;
  	f[idx].y /= g;
}

__global__
void multiplyElementsHPsiK
(
	wfStruct psi,
	hamiltonianStruct hamiltonian,
	paramsStruct params
)
{

  double a, b;

  int i = (blockDim.x * blockIdx.x) + threadIdx.x;
  int j = (blockDim.y * blockIdx.y) + threadIdx.y;
  int idx = i + j * N_x;

  a = psi.d_k[idx].x * hamiltonian.d_T[idx].x - psi.d_k[idx].y * hamiltonian.d_T[idx].y;
  b = psi.d_k[idx].x * hamiltonian.d_T[idx].y + psi.d_k[idx].y * hamiltonian.d_T[idx].x;

  psi.d_k[idx].x = a * exp(- params.d_abc[i] - params.d_abc[j]);
  psi.d_k[idx].y = b * exp(- params.d_abc[i] - params.d_abc[j]);
}

void h_timePropagation (
	wfStruct psi,
	hamiltonianStruct hamiltonian,
	observablesStruct observables,
	potentialStruct potential,
	paramsStruct params,
	tempStruct temp,
	handleStruct handles
)
{
	dim3 blocks (N_x / MTPB, N_x / MTPB, 1);
	dim3 thread (MTPB, MTPB, 1);

	uint tdx;

	cufftHandle plan;
    cufftPlan2d(& plan, N_x, N_x, CUFFT_Z2Z);

	printf("\n  4. Time Propagation \n\n");

	for (tdx = 0; tdx < N_t; tdx++)
	{
		if (tdx == 0)
		{
			multiplyElementsHPsiPHalfStep <<< blocks, thread >>> (psi, hamiltonian, params, tdx);
	    	cudaDeviceSynchronize();
		}

	    fftShift2D <<< blocks, thread >>> (psi.d_x);
	    cudaDeviceSynchronize();

	    cufftExecZ2Z(plan, psi.d_x, psi.d_k, CUFFT_FORWARD);
	    cudaDeviceSynchronize();

		multiplyElementsHPsiK <<< blocks, thread >>> (psi, hamiltonian, params);
		cudaDeviceSynchronize();

	    cufftExecZ2Z(plan, psi.d_k, psi.d_x, CUFFT_INVERSE);
	    cudaDeviceSynchronize();

	    fftShift2D <<< blocks, thread >>> (psi.d_k);
	    cudaDeviceSynchronize();

	    fftShift2D <<< blocks, thread >>> (psi.d_x);
	    cudaDeviceSynchronize();

		if (tdx == N_t - 1)
		{
			multiplyElementsHPsiPHalfStep <<< blocks, thread >>> (psi, hamiltonian, params, tdx);
	    	cudaDeviceSynchronize();
		}
		else
		{
			multiplyElementsHPsiP <<< blocks, thread >>> (psi, hamiltonian, params, tdx);
	    	cudaDeviceSynchronize();
		}

		// multiplyElementsInt <<< blocks, thread >>> (psi, params, tdx);
	    // cudaDeviceSynchronize();

		if (tdx > 0)
	    {
			divideElements <<< blocks, thread >>> (psi.d_x, (double) N_x * N_x);
			divideElements <<< blocks, thread >>> (psi.d_k, (double) N_x * N_x);
		}

		h_getNorm (psi, params, temp, handles, tdx);
		h_calculateDipoleAcceleration (psi, params, potential, temp, observables, handles, tdx);

		if (tdx % 100 == 0)
		{
			printf("     Time advancement calculation progress: %.3lf fs / %.3lf fs | %.3E \n", tdx * dt / attosecondAU / 1000.0, N_t * dt / attosecondAU / 1000.0, psi.h_norm[tdx]);
			if (tdx < N_t - 1)
	    	{
				printf ("\033[1A");
				printf ("\033[K");
			}
		}
		else if (tdx == N_t - 1)
		{
			printf("     Time advancement calculation progress: %.3lf fs / %.3lf fs | %.3E \n", tdx * dt / attosecondAU / 1000.0, N_t * dt / attosecondAU / 1000.0, psi.h_norm[tdx]);
		}
	}
}
