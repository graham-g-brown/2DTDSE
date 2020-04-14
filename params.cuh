#define CUDA_DEVICE (0) 


#define attosecondAU (4.13413746E-02) 
#define SIM_DATE (20200414)  
#define SIM_IDX  (0)  

#define MTPB (32) 

#define GS_CALCULATION (1) 
#define TD_CALCULATION (0) 

#define N_x (1024)   
#define dx (6.25000000E-02)  
#define dk (9.81747704E-02)  

#define ABC_WIDTH (128)   
#define ABC_STRENGTH (1.00000000E-01)   
#define ABC_START (5.00000000E+01)   

#define N_t (8192)   
#define dt (4.13413746E-02)  

#define convergenceFactor (1.00000000E-14)  
#define alpha (5.00000000E-01)  
#define beta (3.38724000E-01)  

#define I0 (2.84981476E-02)  
#define I1 (0.00000000E+00)  
#define E0 (1.68813944E-01)  
#define E1 (0.00000000E+00)  
#define tau0 (6.20120619E+01)  
#define tau1 (2.89389622E+02)  
#define t0 (8.46671351E+01)  
#define t1 (8.46671351E+01)  
#define omega0 (5.69614206E-02)  
#define omega1 (5.69614206E-02)  
#define cep0 (1.57079633E+00)  
#define cep1 (0.00000000E+00)  

#define TD_SIMULATION (0) 
#define WRITE_GS_PARAMS (1) 
#define WRITE_GS_POTENTIAL (1) 
#define WRITE_GS_WAVE_FUNCTION (1) 
#define WRITE_GS_HAMILTONIAN (1) 
#define NEW_SIMULATION (1) 

#define LOCAL_GS_PARAMS_X + "../output/20200414/00000/parameters/x.bin"
#define LOCAL_GS_PARAMS_K + "../output/20200414/00000/parameters/k.bin"
#define LOCAL_GS_PARAMS_T + "../output/20200414/00000/parameters/t.bin"
#define LOCAL_GS_PARAMS_E + "../output/20200414/00000/parameters/E.bin"
#define LOCAL_GS_PARAMS_ABC + "../output/20200414/00000/parameters/abc.bin"

#define LOCAL_GS_POTENTIAL + "../output/20200414/00000/groundState/potential.bin"

#define LOCAL_GS_WAVE_FUNCTION_X + "../output/20200414/00000/groundState/psi_x.bin"
#define LOCAL_GS_WAVE_FUNCTION_K + "../output/20200414/00000/groundState/psi_k.bin"

#define LOCAL_GS_HAMILTONIAN_T + "../output/20200414/00000/groundState/hamiltonianT.bin"
#define LOCAL_GS_HAMILTONIAN_V + "../output/20200414/00000/groundState/hamiltonianV.bin"

#define LOCAL_GS_HAMILTONIAN_IT + "../output/20200414/00000/groundState/hamiltonianiT.bin"
#define LOCAL_GS_HAMILTONIAN_IV + "../output/20200414/00000/groundState/hamiltonianiV.bin"

#define LOCAL_TD_T + "../output/20200414/00000/timeDependent/t.bin"
#define LOCAL_TD_E + "../output/20200414/00000/timeDependent/E.bin"

#define LOCAL_TD_OBSERVABLES_DIPOLE_ACCELERATION + "../output/20200414/00000/timeDependent/observables/dipoleAcceleration.bin"
#define LOCAL_TD_OBSERVABLES_NORM + "../output/20200414/00000/timeDependent/observables/norm.bin"

#include "./subFunctions/system/h_checkCUDADevice.cuh"
#include "./subFunctions/initialisation/h_initialiseSimulation.cuh"
#include "./subFunctions/output/h_writeGSData.cuh"
#include "./subFunctions/output/h_writeTDData.cuh"

#include "./subFunctions/groundState/h_initialiseGroundStateCalculation.cuh"
#include "./subFunctions/timeDependent/observables/h_calculateDipoleAcceleration.cuh"
#include "./subFunctions/timeDependent/h_timePropagation.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
