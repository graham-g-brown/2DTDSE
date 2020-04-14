__global__
void d_prepDipoleAcceleration (
	wfStruct psi,
	potentialStruct potential,
	tempStruct temp
)
{
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	int j = (blockDim.y * blockIdx.y) + threadIdx.y;
  	int idx = i + j * N_x;

	temp.d_Z1[idx].x = potential.d_dV[idx] * psi.d_x[idx].x;
	temp.d_Z1[idx].y = potential.d_dV[idx] * psi.d_x[idx].y;
}

void h_calculateDipoleAcceleration (
	wfStruct psi,
	paramsStruct params,
	potentialStruct potential,
	tempStruct temp,
	observablesStruct observables,
	handleStruct handles,
	int tdx
)
{
	dim3 blocks (N_x / MTPB, N_x / MTPB, 1);
	dim3 thread (MTPB, MTPB, 1);

	d_prepDipoleAcceleration <<< blocks, thread >>> (psi, potential, temp);

	cublasZdotc(handles.h_blas, N_x * N_x, psi.d_x, 1, temp.d_Z1, 1, & observables.h_dipoleAcceleration[tdx]);
}
