void h_writeGSParams (
	paramsStruct params
)
{
	FILE * IO;

	IO = fopen(LOCAL_GS_PARAMS_X, "wb");
    fwrite(params.h_x, sizeof(double), N_x, IO);
    fclose(IO);

	IO = fopen(LOCAL_GS_PARAMS_K, "wb");
    fwrite(params.h_x, sizeof(double), N_x, IO);
    fclose(IO);

	IO = fopen(LOCAL_GS_PARAMS_ABC, "wb");
    fwrite(params.h_abc, sizeof(double), N_x, IO);
    fclose(IO);

	if (TD_SIMULATION == 1)
	{
		IO = fopen(LOCAL_GS_PARAMS_T, "wb");
	    fwrite(params.h_t, sizeof(double), N_t, IO);
	    fclose(IO);

		IO = fopen(LOCAL_GS_PARAMS_E, "wb");
	    fwrite(params.h_E, sizeof(double), N_t, IO);
	    fclose(IO);
	}
}

void h_writeGSPotential (
	potentialStruct potential
)
{
	FILE * IO;

	IO = fopen(LOCAL_GS_POTENTIAL, "wb");
    fwrite(potential.h_V, sizeof(double), N_x * N_x, IO);
    fclose(IO);
}

void h_writeGSWaveFunction (
	wfStruct psi
)
{
	FILE * IO;

	IO = fopen(LOCAL_GS_WAVE_FUNCTION_X, "wb");
    fwrite(psi.h_x, sizeof(double2), N_x * N_x, IO);
    fclose(IO);

	IO = fopen(LOCAL_GS_WAVE_FUNCTION_K, "wb");
    fwrite(psi.h_k, sizeof(double2), N_x * N_x, IO);
    fclose(IO);
}

void h_writeGSHamiltonian (
	hamiltonianStruct hamiltonian
)
{
	FILE * IO;

	IO = fopen(LOCAL_GS_HAMILTONIAN_T, "wb");
    fwrite(hamiltonian.h_T, sizeof(double2), N_x * N_x, IO);
    fclose(IO);

	IO = fopen(LOCAL_GS_HAMILTONIAN_V, "wb");
    fwrite(hamiltonian.h_V, sizeof(double2), N_x * N_x, IO);
    fclose(IO);

	IO = fopen(LOCAL_GS_HAMILTONIAN_IT, "wb");
    fwrite(hamiltonian.h_iT, sizeof(double), N_x * N_x, IO);
    fclose(IO);

	IO = fopen(LOCAL_GS_HAMILTONIAN_IV, "wb");
    fwrite(hamiltonian.h_iV, sizeof(double), N_x * N_x, IO);
    fclose(IO);
}

void h_writeGSData (
	paramsStruct params,
	potentialStruct potential,
	wfStruct psi,
	hamiltonianStruct hamiltonian
)
{
	cudaMemcpy (psi.h_x, psi.d_x, N_x * N_x * sizeof(double2), cudaMemcpyDeviceToHost);
	cudaMemcpy (psi.h_k, psi.d_k, N_x * N_x * sizeof(double2), cudaMemcpyDeviceToHost);

	if (WRITE_GS_PARAMS == 1)
	{
		h_writeGSParams (params);
	}

	if (WRITE_GS_POTENTIAL == 1)
	{
		h_writeGSPotential (potential);
	}

	if (WRITE_GS_WAVE_FUNCTION == 1)
	{
		h_writeGSWaveFunction (psi);
	}

	if (WRITE_GS_HAMILTONIAN == 1)
	{
		h_writeGSHamiltonian (hamiltonian);
	}
}
