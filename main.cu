// C Libraries
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// CUDA Libraries
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <ctime>
#include <cstdio>
#include <cusolverDn.h>
#include <complex>

//Type Definition for Complex-Type
using namespace std;
typedef complex <double> long_double_complex;
long_double_complex Il = long_double_complex(0.0,1.0);

typedef struct
{
	double2 * h_T;
	double2 * h_V;
	double  * h_iT;
	double  * h_iV;

	double2 * d_T;
	double2 * d_V;
	double  * d_iT;
	double  * d_iV;

} hamiltonianStruct;

typedef struct
{
	double * h_x;
	double * h_k;
	double * h_t;
	double * h_E;
	double * h_abc;

	double * d_x;
	double * d_k;
	double * d_t;
	double * d_E;
	double * d_abc;

} paramsStruct;

typedef struct
{
	double2 * h_x;
	double2 * h_k;
	double  * h_norm;

	double2 * d_x;
	double2 * d_k;
	double  * d_norm;

} wfStruct;

typedef struct
{
	double * h_V;
	double * h_dV;
	double * d_V;
	double * d_dV;

} potentialStruct;

typedef struct
{
	double  * d_D1;
	double2 * d_Z1;

} tempStruct;

typedef struct
{
	double2 * h_dipoleAcceleration;

	double2 * d_dipoleAcceleration;

} observablesStruct;

typedef struct
{
	cublasHandle_t h_blas;
	cublasHandle_t d_blas;

} handleStruct;

#include "params.cuh"

int main (int argc, char * argv[])
{
	// Check CUDA device and print computation parameters

	h_checkCUDADevice ();

	//------------------------//
	// POINTER INITIALISATION //
	//------------------------//

	// Initialise Parameter Pointers
	//
	// - h_x : host spatial array
	// - h_k : host momentum array
	// - h_t : host time array
	// - h_E : host electric field array
	//
	// - d_x : device spatial array
	// - d_k : device momentum array
	// - d_t : device time array
	// - d_E : device electric field array

	paramsStruct params;

	params.h_x   = (double *) malloc (N_x * sizeof(double));
	params.h_k   = (double *) malloc (N_x * sizeof(double));
	params.h_t   = (double *) malloc (N_t * sizeof(double));
	params.h_E   = (double *) malloc (N_t * sizeof(double));
	params.h_abc = (double *) malloc (N_x * sizeof(double));

	cudaMalloc ((void **) & params.d_x  , N_x * sizeof(double));
	cudaMalloc ((void **) & params.d_k  , N_x * sizeof(double));
	cudaMalloc ((void **) & params.d_t  , N_t * sizeof(double));
	cudaMalloc ((void **) & params.d_E  , N_t * sizeof(double));
	cudaMalloc ((void **) & params.d_abc, N_x * sizeof(double));

	// Initialise Potential Pointers
	//
	// h_V  : host potential matrix
	// h_dV : host potential gradient matrix
	//
	// d_V  : device potential matrix
	// d_dV : device potential gradient matrix

	potentialStruct potential;

	potential.h_V = (double *) malloc (N_x * N_x * sizeof(double));
	potential.h_dV = (double *) malloc (N_x * N_x * sizeof(double));

	cudaMalloc ((void **) & potential.d_V, N_x * N_x * sizeof(double));
	cudaMalloc ((void **) & potential.d_dV, N_x * N_x * sizeof(double));

	// Initialise Wave Function Pointers
	//
	// h_x : host position-space wave function
	// h_k : host momentum-space wave function
	//
	// d_x : device position-space wave function
	// d_k : device momentum-space wave function

	wfStruct psi;

	psi.h_x    = (double2 *) malloc (N_x * N_x * sizeof(double2));
	psi.h_k    = (double2 *) malloc (N_x * N_x * sizeof(double2));
	psi.h_norm = (double *) malloc (N_t * sizeof(double));

	cudaMalloc ((void **) & psi.d_x, N_x * N_x * sizeof(double2));
	cudaMalloc ((void **) & psi.d_k, N_x * N_x * sizeof(double2));
	cudaMalloc ((void **) & psi.d_norm, N_t * sizeof(double));

	// Initialise Hamiltonian Pointers
	//
	// h_T  : host real-time kinetic energy Hamiltonian
	// h_V  : host real-time potential energy Hamiltonian
	// h_iT : host imaginary-time kinetic energy Hamiltonian
	// h_iV : host imaginary-time potential energy Hamiltonian
	//
	// d_T  : device real-time kinetic energy Hamiltonian
	// d_V  : device real-time potential energy Hamiltonian
	// d_iT : device imaginary-time kinetic energy Hamiltonian
	// d_iV : device imaginary-time potential energy Hamiltonian

	hamiltonianStruct hamiltonian;

	hamiltonian.h_T = (double2 *) malloc (N_x * N_x * sizeof(double2));
	hamiltonian.h_V = (double2 *) malloc (N_x * N_x * sizeof(double2));
	hamiltonian.h_iT = (double *) malloc (N_x * N_x * sizeof(double));
	hamiltonian.h_iV = (double *) malloc (N_x * N_x * sizeof(double));

	cudaMalloc ((void **) & hamiltonian.d_T, N_x * N_x * sizeof(double2));
	cudaMalloc ((void **) & hamiltonian.d_V, N_x * N_x * sizeof(double2));
	cudaMalloc ((void **) & hamiltonian.d_iT, N_x * N_x * sizeof(double));
	cudaMalloc ((void **) & hamiltonian.d_iV, N_x * N_x * sizeof(double));

	// Initialise Temp Pointers
	//
	// d_D1 : device double-valued matrix
	// d_Z1 : device complex-valued matrix

	tempStruct temp;

	cudaMalloc ((void **) & temp.d_D1, N_x * N_x * sizeof(double));
	cudaMalloc ((void **) & temp.d_Z1, N_x * N_x * sizeof(double2));

	// Initialise Observables Pointers
	//
	// h_dipoleAcceleration : host dipole acceleration array
	//
	// d_dipoleAcceleration : device dipole acceleration array

	observablesStruct observables;

	observables.h_dipoleAcceleration = (double2 *) malloc (N_t * sizeof(double2));

	cudaMalloc ((void **) & observables.d_dipoleAcceleration, N_t * sizeof(double2));

	// Initialise Handles (cuBLAS)
	//
	// h_blas : host-oriented cuBLAS handle
	//
	// d_blas : device-oriented cuBLAS handle

	handleStruct handles;

	cublasCreate (& handles.h_blas);
	cublasSetPointerMode(handles.h_blas, CUBLAS_POINTER_MODE_HOST);
	cublasCreate (& handles.d_blas);
	cublasSetPointerMode(handles.d_blas, CUBLAS_POINTER_MODE_DEVICE);

	//------------------//
	// BEGIN SIMULATION //
	//------------------//

	// Initialise parameter values
	// - mesh, time-dependent, etc...

	h_initialiseSimulation (params, potential, psi, hamiltonian);

	// Calculate ground state
	// Imaginary time-propagation

	h_initialiseGroundStateCalculation (psi, hamiltonian, params, temp, handles);

	// Write ground state and parameter files

	h_writeGSData (params, potential, psi, hamiltonian);

	if (TD_SIMULATION == 1)
	{
		// Time propagation
		// Fourier split-step method

		h_timePropagation (psi, hamiltonian, observables, potential, params, temp, handles);

		// Write time-dependent data

		h_writeTDData (params, observables, psi);
	}

	// Free Pointers

	free (params.h_x);
	free (params.h_k);
	free (params.h_t);
	free (params.h_E);
	free (params.h_abc);

	free (potential.h_V);
	free (potential.h_dV);

	free (psi.h_x);
	free (psi.h_k);
	free (psi.h_norm);

	free (hamiltonian.h_T);
	free (hamiltonian.h_V);
	free (hamiltonian.h_iT);
	free (hamiltonian.h_iV);

	free (observables.h_dipoleAcceleration);

	cudaFree (params.d_x);
	cudaFree (params.d_k);
	cudaFree (params.d_t);
	cudaFree (params.d_E);
	cudaFree (params.d_abc);

	cudaFree (potential.d_V);
	cudaFree (potential.d_dV);

	cudaFree (psi.d_x);
	cudaFree (psi.d_k);
	cudaFree (psi.d_norm);

	cudaFree (hamiltonian.d_T);
	cudaFree (hamiltonian.d_V);
	cudaFree (hamiltonian.d_iT);
	cudaFree (hamiltonian.d_iV);

	cudaFree (temp.d_D1);
	cudaFree (temp.d_Z1);

	cudaFree (observables.d_dipoleAcceleration);

	cublasDestroy(handles.h_blas);
	cublasDestroy(handles.d_blas);
}
