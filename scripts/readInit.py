import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from datetime import date
import os
from sys import platform
import sys

COLOR = 'white'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["blue","black","red"])

SMALL_SIZE = 27
MEDIUM_SIZE = 30
BIGGER_SIZE = 40
FIG_DPI=100

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.switch_backend('agg')

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

attosecond_au   = 0.0413413745758
femtosecond_au  = 41.3413745758
c_au            = 137.035999074
intensity_au    = 1.0 / 3.509E16
electronvolt_au = 1.0 / 27.2114

with open("init") as f:
    for line in f:
        exec(line)

try:
	GS_CALCULATION
except NameError:
	GS_CALCULATION = True

try:
	TD_CALCULATION
except NameError:
	TD_CALCULATION = False

try:
	N_x
except NameError:
	N_x = 1024

try:
	N_t
except NameError:
	N_t = 16384

try:
	dt
except NameError:
	dt = 1.0 * attosecond_au

try:
	convergenceFactor
except NameError:
	convergenceFactor = 1E-14

try:
	alpha
except NameError:
	alpha = 0.50

try:
	beta
except NameError:
	beta = 0.338724

try:
	dx
except NameError:
	dx = 0.0625

try:
	dk
except NameError:
	dk = 2.0 * np.pi / dx / N_x

try:
	I0
except NameError:
	I0 = 5.0E14 * intensity_au

try:
	I1
except NameError:
	I1 = 0.0 * intensity_au

try:
	tau0
except NameError:
	tau0 = 7.00 * femtosecond_au

try:
	tau1
except NameError:
	tau1 = 7.0 * femtosecond_au

try:
	t0
except NameError:
	t0 = N_t * dt / 4.0

try:
	t1
except NameError:
	t1 = N_t * dt / 4.0

try:
	omega0
except NameError:
	omega0 = 1.55 * electronvolt_au

try:
	omega1
except NameError:
	omega1 = 1.55 * electronvolt_au

try:
	TD_SIMULATION
except NameError:
	TD_SIMULATION = 0

try:
	WRITE_GS_PARAMS
except NameError:
	WRITE_GS_PARAMS = 1

try:
	WRITE_GS_POTENTIAL
except NameError:
	WRITE_GS_POTENTIAL = 1

try:
	WRITE_GS_WAVE_FUNCTION
except NameError:
	WRITE_GS_WAVE_FUNCTION = 1

try:
	NEW_SIMULATION
except NameError:
	NEW_SIMULATION = 1

try:
	WRITE_GS_HAMILTONIAN
except NameError:
	WRITE_GS_HAMILTONIAN = 1

try:
	ABC_WIDTH
except NameError:
	ABC_WIDTH = N_x // 8

try:
	ABC_STRENGTH
except NameError:
	ABC_STRENGTH = 0.1

try:
	ABC_START
except NameError:
	ABC_START = 50.0

def welcomeScreen():

	print("")
	print(color.BOLD + "  GGB - 1D Helium" + color.END)
	print(color.BOLD + "  One-Dimensional Two-Electron Exact TDSE Simulation  " + color.END)
	print("")
	print("  A one-dimensional two-electron simulation using the Fourier")
	print("  split-step propagation scheme.")
	print("")

def printTargetParameters(N_x, dx, N_t, dt, I0, tau0, omega0, I1, tau1, omega1):

	print("")
	print("  1. Simulation Parameters")
	print("")
	print("     Charge = 2")
	print("")
	print("     N_x    = %d" % N_x)
	print("     dx     = %.2lf a.u." % dx)
	print("     L      = %.2lf a.u." % (N_x * dx))
	print("")
	print("     N_t    = %d" % (N_t))
	print("     dt     = %.2lf a.u." % dt)
	print("     T      = %.2lf a.u." % (N_t * dt))
	print("")
	print("     I0     = %.2E a.u." % (I0))
	print("     tau0   = %.2lf a.u." % (tau0))
	print("     omega0 = %.2E a.u." % (omega0))
	print("     I1     = %.2E a.u." % (I1))
	print("     tau1   = %.2lf a.u." % (tau1))
	print("     omega1 = %.2E a.u." % (omega1))
	print("")

date = date.today()
date_today = int(str(int(date.year)) + str(int(date.month)).zfill(2) + str(int(date.day)).zfill(2))

E0 = np.sqrt(I0 / 3.509E16)
E1 = np.sqrt(I1 / 3.509E16)

localOutputFilePath = "../output"
idx = 0
con = True

localDateFilePath = localOutputFilePath + "/" + str(date_today)

while (con):
	if not os.path.exists(localOutputFilePath + "/" + str(date_today) + "/" + str(idx).zfill(5)):
		con = False
	else:
		idx += 1

localSimIndexFilePath = localOutputFilePath + "/" + str(date_today) + "/" + str(idx).zfill(5)
localParametersFilePath = localSimIndexFilePath + "/parameters"
localGroundStateFilePath = localSimIndexFilePath + "/groundState"
localTimeDependentFilePath = localSimIndexFilePath + "/timeDependent"
localObservablesFilePath = localTimeDependentFilePath + "/observables"

path = localDateFilePath
if not os.path.exists(path):
	os.mkdir(path)

path = localSimIndexFilePath
if not os.path.exists(path):
	os.mkdir(path)

path = localParametersFilePath
if not os.path.exists(path):
	os.mkdir(path)

path = localGroundStateFilePath
if not os.path.exists(path):
	os.mkdir(path)

path = localTimeDependentFilePath
if not os.path.exists(path):
	os.mkdir(path)

path = localObservablesFilePath
if not os.path.exists(path):
	os.mkdir(path)

LOCAL_GS_PARAMS_X = localParametersFilePath + "/x.bin"
LOCAL_GS_PARAMS_K = localParametersFilePath + "/k.bin"
LOCAL_GS_PARAMS_T = localParametersFilePath + "/t.bin"
LOCAL_GS_PARAMS_E = localParametersFilePath + "/E.bin"
LOCAL_GS_PARAMS_ABC = localParametersFilePath + "/abc.bin"

LOCAL_GS_POTENTIAL = localGroundStateFilePath + "/potential.bin"

LOCAL_GS_WAVE_FUNCTION_X = localGroundStateFilePath + "/psi_x.bin"
LOCAL_GS_WAVE_FUNCTION_K = localGroundStateFilePath + "/psi_k.bin"

LOCAL_GS_HAMILTONIAN_T = localGroundStateFilePath + "/hamiltonianT.bin"
LOCAL_GS_HAMILTONIAN_V = localGroundStateFilePath + "/hamiltonianV.bin"

LOCAL_GS_HAMILTONIAN_IT = localGroundStateFilePath + "/hamiltonianiT.bin"
LOCAL_GS_HAMILTONIAN_IV = localGroundStateFilePath + "/hamiltonianiV.bin"

LOCAL_TD_T = localTimeDependentFilePath + "/t.bin"
LOCAL_TD_E = localTimeDependentFilePath + "/E.bin"

LOCAL_TD_OBSERVABLES_DIPOLE_ACCELERATION = localTimeDependentFilePath + "/observables/dipoleAcceleration.bin"
LOCAL_TD_OBSERVABLES_NORM = localTimeDependentFilePath + "/observables/norm.bin"

def writeParametersFile(filepath):
	f = open(filepath, "w")
	f.write("#define CUDA_DEVICE (0) \n")
	f.write("\n")
	f.write("\n")
	f.write("#define attosecondAU (%.8E) \n" % attosecond_au)
	f.write("#define SIM_DATE (%d)  \n" % date_today)
	f.write("#define SIM_IDX  (%d)  \n" % idx)
	f.write("\n")
	f.write("#define MTPB (32) \n")
	f.write("\n")
	f.write("#define GS_CALCULATION (%d) \n" % GS_CALCULATION)
	f.write("#define TD_CALCULATION (%d) \n" % TD_CALCULATION)
	f.write("\n")
	f.write("#define N_x (%d)   \n" % N_x)
	f.write("#define dx (%.8E)  \n" % dx)
	f.write("#define dk (%.8E)  \n" % dk)
	f.write("\n")
	f.write("#define ABC_WIDTH (%d)   \n" % ABC_WIDTH)
	f.write("#define ABC_STRENGTH (%.8E)   \n" % ABC_STRENGTH)
	f.write("#define ABC_START (%.8E)   \n" % ABC_START)
	f.write("\n")
	f.write("#define N_t (%d)   \n" % N_t)
	f.write("#define dt (%.8E)  \n" % dt)
	f.write("\n")
	f.write("#define convergenceFactor (%.8E)  \n" % convergenceFactor)
	f.write("#define alpha (%.8E)  \n" % alpha)
	f.write("#define beta (%.8E)  \n" % beta)
	f.write("\n")
	f.write("#define I0 (%.8E)  \n" % I0)
	f.write("#define I1 (%.8E)  \n" % I1)
	f.write("#define E0 (%.8E)  \n" % (np.sqrt(I0)))
	f.write("#define E1 (%.8E)  \n" % (np.sqrt(I1)))
	f.write("#define tau0 (%.8E)  \n" % tau0)
	f.write("#define tau1 (%.8E)  \n" % tau1)
	f.write("#define t0 (%.8E)  \n" % t0)
	f.write("#define t1 (%.8E)  \n" % t1)
	f.write("#define omega0 (%.8E)  \n" % omega0)
	f.write("#define omega1 (%.8E)  \n" % omega1)
	f.write("#define cep0 (%.8E)  \n" % (np.pi / 2.0))
	f.write("#define cep1 (%.8E)  \n" % (0))
	f.write("\n")
	f.write("#define TD_SIMULATION (%d) \n" % (TD_SIMULATION))
	f.write("#define WRITE_GS_PARAMS (%d) \n" % (WRITE_GS_PARAMS))
	f.write("#define WRITE_GS_POTENTIAL (%d) \n" % (WRITE_GS_POTENTIAL))
	f.write("#define WRITE_GS_WAVE_FUNCTION (%d) \n" % (WRITE_GS_WAVE_FUNCTION))
	f.write("#define WRITE_GS_HAMILTONIAN (%d) \n" % (WRITE_GS_HAMILTONIAN))
	f.write("#define NEW_SIMULATION (%d) \n" % (NEW_SIMULATION))
	f.write("\n")
	f.write("#define LOCAL_GS_PARAMS_X + \"" + LOCAL_GS_PARAMS_X + "\"\n")
	f.write("#define LOCAL_GS_PARAMS_K + \"" + LOCAL_GS_PARAMS_K + "\"\n")
	f.write("#define LOCAL_GS_PARAMS_T + \"" + LOCAL_GS_PARAMS_T + "\"\n")
	f.write("#define LOCAL_GS_PARAMS_E + \"" + LOCAL_GS_PARAMS_E + "\"\n")
	f.write("#define LOCAL_GS_PARAMS_ABC + \"" + LOCAL_GS_PARAMS_ABC + "\"\n")
	f.write("\n")
	f.write("#define LOCAL_GS_POTENTIAL + \"" + LOCAL_GS_POTENTIAL + "\"\n")
	f.write("\n")
	f.write("#define LOCAL_GS_WAVE_FUNCTION_X + \"" + LOCAL_GS_WAVE_FUNCTION_X + "\"\n")
	f.write("#define LOCAL_GS_WAVE_FUNCTION_K + \"" + LOCAL_GS_WAVE_FUNCTION_K + "\"\n")
	f.write("\n")
	f.write("#define LOCAL_GS_HAMILTONIAN_T + \"" + LOCAL_GS_HAMILTONIAN_T + "\"\n")
	f.write("#define LOCAL_GS_HAMILTONIAN_V + \"" + LOCAL_GS_HAMILTONIAN_V + "\"\n")
	f.write("\n")
	f.write("#define LOCAL_GS_HAMILTONIAN_IT + \"" + LOCAL_GS_HAMILTONIAN_IT + "\"\n")
	f.write("#define LOCAL_GS_HAMILTONIAN_IV + \"" + LOCAL_GS_HAMILTONIAN_IV + "\"\n")
	f.write("\n")
	f.write("#define LOCAL_TD_T + \"" + LOCAL_TD_T + "\"\n")
	f.write("#define LOCAL_TD_E + \"" + LOCAL_TD_E + "\"\n")
	f.write("\n")
	f.write("#define LOCAL_TD_OBSERVABLES_DIPOLE_ACCELERATION + \"" + LOCAL_TD_OBSERVABLES_DIPOLE_ACCELERATION + "\"\n")
	f.write("#define LOCAL_TD_OBSERVABLES_NORM + \"" + LOCAL_TD_OBSERVABLES_NORM + "\"\n")
	f.write("\n")
	f.write("#include \"./subFunctions/system/h_checkCUDADevice.cuh\"\n")
	f.write("#include \"./subFunctions/initialisation/h_initialiseSimulation.cuh\"\n")
	f.write("#include \"./subFunctions/output/h_writeGSData.cuh\"\n")
	f.write("#include \"./subFunctions/output/h_writeTDData.cuh\"\n")
	f.write("\n")
	f.write("#include \"./subFunctions/groundState/h_initialiseGroundStateCalculation.cuh\"\n")
	f.write("#include \"./subFunctions/timeDependent/observables/h_calculateDipoleAcceleration.cuh\"\n")
	f.write("#include \"./subFunctions/timeDependent/h_timePropagation.cuh\"\n")
	f.write("\n")
	f.write("#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }\n")
	f.write("inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)\n")
	f.write("{\n")
	f.write("   if (code != cudaSuccess)\n")
	f.write("   {\n")
	f.write("      fprintf(stderr,\"GPUassert: %s %s %d\\n\", cudaGetErrorString(code), file, line);\n")
	f.write("      if (abort) exit(code);\n")
	f.write("   }\n")
	f.write("}\n")

writeParametersFile("./params.cuh")

welcomeScreen ()
printTargetParameters (N_x, dx, N_t, dt, I0, tau0, omega0, I1, omega1, I1)
