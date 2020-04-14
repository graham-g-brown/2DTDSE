void h_initialiseSimulation (
	paramsStruct params,
	potentialStruct potential,
	wfStruct psi,
	hamiltonianStruct hamiltonian
)
{
	uint i, j, idx;

	double kineticEnergy, potentialEnergy;

	// Initialise one-dimensional arrays
	// - position, momentum, time, electric field

	for (i = 0; i < N_x; i++)
	{
		params.h_x[i] = - 0.50 * double(N_x) * dx + double(i) * dx;
		params.h_k[i] = - M_PI / dx + double(i) * 2.0 * M_PI / double(N_x) / dx;

		if (params.h_x[i] < - ABC_START)
		{
			params.h_abc[i] = ABC_STRENGTH * pow(params.h_x[i] + ABC_START, 2.0);
		}
		else if (params.h_x[i] > ABC_START)
		{
			params.h_abc[i] = ABC_STRENGTH * pow(params.h_x[i] - ABC_START, 2.0);
		}
		else
		{
			params.h_abc[i] = 0.0;
		}
	}

	// for (i = 0; i < ABC_WIDTH; i++)
	// {
	// 	params.h_abc[N_x - ABC_WIDTH + i] = ABC_STRENGTH * pow(double(i), 2.0);
	// 	params.h_abc[ABC_WIDTH - 1 - i]   = ABC_STRENGTH * pow(double(i), 2.0);
	// }

	for (i = 0; i < N_t; i++)
	{
		params.h_t[i] = double(i) * dt - 0.50 * N_t * dt;
		params.h_E[i] = E0 * exp(- pow(params.h_t[i] / (0.5 * tau0), 2.0)) * cos(omega0 * params.h_t[i] + cep0)
					  + E1 * exp(- pow(params.h_t[i] / (0.5 * tau1), 2.0)) * cos(omega1 * params.h_t[i] + cep1);
	}

	// Initialise two-dimensional arrays
	// potential, potential gradient, wave function, hamiltonian

	for (i = 0; i < N_x; i++)
	{
		for (j = 0; j < N_x; j++)
		{
			idx = i + j * N_x;

			potential.h_V[idx]     = 1.0 / sqrt(pow(params.h_x[i] - params.h_x[j], 2.0) + beta)
								   - 2.0 / sqrt(pow(params.h_x[i], 2.0) + alpha)
								   - 2.0 / sqrt(pow(params.h_x[j], 2.0) + alpha);

			potential.h_dV[idx]    = 2.0 * params.h_x[i] / pow(pow(params.h_x[i], 2.0) + alpha, 1.50)
								   + 2.0 * params.h_x[j] / pow(pow(params.h_x[j], 2.0) + alpha, 1.50);

			psi.h_x[idx].x         = exp(- (pow(params.h_x[i], 2.0) + pow(params.h_x[j], 2.0)) / 4.0);
			psi.h_x[idx].y         = 0.0;

			psi.h_k[idx].x         = 0.0;
			psi.h_k[idx].y         = 0.0;

			kineticEnergy   	   = 0.50 * (pow(params.h_k[i], 2.0) + pow(params.h_k[j], 2.0));
			potentialEnergy 	   = potential.h_V[idx];

			hamiltonian.h_T[idx].x = cos(kineticEnergy * dt);
			hamiltonian.h_T[idx].y = - sin(kineticEnergy * dt);

			hamiltonian.h_V[idx].x = potentialEnergy * (0.50 * dt);
			hamiltonian.h_V[idx].y = potentialEnergy * (0.50 * dt);

			hamiltonian.h_iT[idx]  = exp(- kineticEnergy * dt);
			hamiltonian.h_iV[idx]  = exp(- potentialEnergy * (0.50 * dt));
		}
	}

	// Copy host arrays to device

	cudaMemcpy (params.d_x, params.h_x, N_x * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (params.d_k, params.h_k, N_x * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (params.d_t, params.h_t, N_t * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (params.d_E, params.h_E, N_t * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (params.d_abc, params.h_abc, N_x * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy (potential.d_V, potential.h_V, N_x * N_x * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (potential.d_dV, potential.h_dV, N_x * N_x * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy (psi.d_x, psi.h_x, N_x * N_x * sizeof(double2), cudaMemcpyHostToDevice);
	cudaMemcpy (psi.d_k, psi.h_k, N_x * N_x * sizeof(double2), cudaMemcpyHostToDevice);

	cudaMemcpy (hamiltonian.d_T, hamiltonian.h_T, N_x * N_x * sizeof(double2), cudaMemcpyHostToDevice);
	cudaMemcpy (hamiltonian.d_V, hamiltonian.h_V, N_x * N_x * sizeof(double2), cudaMemcpyHostToDevice);

	cudaMemcpy (hamiltonian.d_iT, hamiltonian.h_iT, N_x * N_x * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (hamiltonian.d_iV, hamiltonian.h_iV, N_x * N_x * sizeof(double), cudaMemcpyHostToDevice);
}
