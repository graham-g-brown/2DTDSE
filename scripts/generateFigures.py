import numpy as np
import numpy.linalg as nla
import matplotlib.pylab as plt
from importlib.machinery import SourceFileLoader
import os
import scipy.special as sps
import scipy.interpolate as sin
import matplotlib as mpl
import matplotlib.colors

def padFunction(f, scale=8, pad=0.0):

	N = np.size(f)

	N_new = scale * N

	f_new = np.ones(N_new, dtype=f.dtype) * pad

	f_new[N_new // 2 - N // 2 : N_new // 2 + N // 2] = f[:]

	return f_new

def applyFourierTransformWindow (f):

	N = np.size(f)

	idx = np.linspace(- 0.5 * N, 0.5 * N - 1, N)

	filter = np.exp(- np.power(idx / (N // 4), 2.0))

	return f * filter

def contourPlot (f, X, Y, xMindx=0, xMaxdx=-1, path=None, xlabel=None, ylabel=None, figsize=(14,12)):
	fmax = np.max(np.absolute(f))
	figure = plt.figure(figsize=figsize)
	plt.pcolormesh(X[xMindx : xMaxdx, xMindx : xMaxdx],\
	               Y[xMindx : xMaxdx, xMindx : xMaxdx], \
	               f[xMindx : xMaxdx, xMindx : xMaxdx] / fmax, cmap=cmap, vmin = - 1.0, vmax = 1.0)
	plt.colorbar(format='%.1f')
	plt.xlabel(r'$x$ (a.u.)')
	plt.ylabel(r'$y$ (a.u.)')
	if (xlabel is not None):
		plt.xlabel(xlabel)
	if (ylabel is not None):
		plt.ylabel(ylabel)
	if (path is None):
		plt.show()
	else:
		plt.savefig(path, transparent=True, bbox_inches="tight", dpi=FIG_DPI)
	plt.close(figure)

def linearPlot(x, f, xlabel=None, ylabel=None, xmin=None, xmax=None, ymin=None, ymax=None, path=None, log=False):

	if (xmin is None):
		xmin = x[0]
	if (xmax is None):
		xmax = x[-1]
	if (ymin is None):
		if (log):
			ymin = 0.10 * np.min(f)
		else:
			ymin = 1.1 * np.min(f)
	if (ymax is None):
		if (log):
			ymax = 10.0 * np.max(f)
		else:
			ymax = 1.1 * np.max(f)

	figure = plt.figure(figsize=(16,12))
	if (log):
		plt.semilogy(x, f, linewidth=2.0)
		plt.fill_between(x, f, alpha=0.50)
	else:
		plt.plot(x, f, linewidth=2.0)
		plt.fill_between(x, f, alpha=0.50)
	plt.grid()
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	if (xlabel is not None):
		plt.xlabel(xlabel)
	if (ylabel is not None):
		plt.ylabel(ylabel)
	if (path is None):
		plt.show()
	else:
		plt.savefig(path, format="PNG", bbox_inches="tight", transparent=True, dpi=FIG_DPI)
	plt.close(figure)

def logPlot(x, f, xlabel=None, ylabel=None, xmin=None, xmax=None, ymin=None, ymax=None, path=None):

	if (xmin is None):
		xmin = x[0]
	if (xmax is None):
		xmax = x[-1]
	if (ymin is None):
		ymin = 1.1 * np.min(f)
	if (ymax is None):
		ymax = 1.1 * np.max(f)

	figure = plt.figure(figsize=(16,12))
	plt.semilogy(x, f, linewidth=2.0)
	# plt.fill_between(x, f, alpha=0.50)
	plt.grid()
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	if (xlabel is not None):
		plt.xlabel(xlabel)
	if (ylabel is not None):
		plt.ylabel(ylabel)
	if (path is None):
		plt.show()
	else:
		plt.savefig(path, format="PNG", bbox_inches="tight", transparent=True, dpi=FIG_DPI)
	plt.close(figure)

GENERATE_HAMILTONIAN_FIGURE = False

COLOR = 'white'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","black","red"])

SMALL_SIZE = 32
MEDIUM_SIZE = 30
BIGGER_SIZE = 40
FIG_DPI=300

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.switch_backend('agg')

params = SourceFileLoader("params", "./scripts/parseParams.py").load_module()

attosecond_au   = 0.0413413745758
femtosecond_au  = 41.3413745758
c_au            = 137.035999074
intensity_au    = 1.0 / 3.509E16
electronvolt_au = 1.0 / 27.2114

SIM_DATE = int(params.SIM_DATE[1 : -1])
SIM_IDX  = int(params.SIM_IDX[1 : -1])
TD_SIMULATION = int(params.TD_SIMULATION[1 : -1])

N_x = int(params.N_x[1 : -1])
dx  = float(params.dx[1 : -1])

N_t = int(params.N_t[1 : -1])
dt  = float(params.dt[1 : -1])

I0     = float(params.I0[1 : -1])
I1     = float(params.I1[1 : -1])
omega0 = float(params.omega0[1 : -1])
omega1 = float(params.omega1[1 : -1])
tau0   = float(params.tau0[1 : -1])
tau1   = float(params.tau1[1 : -1])

rootFilePath = "../output/" + str(SIM_DATE) + "/" + str(SIM_IDX).zfill(5)

###################
# Load Data Files #
###################

# Define Data Paths

# # Position and Momentum Vectors

xFilePath = rootFilePath + "/parameters/x.bin"
kFilePath = rootFilePath + "/parameters/k.bin"

# # Absorbing Boundary Condition Vector

abcFilePath = rootFilePath + "/parameters/abc.bin"

# # Time-Dependent Vectors

tFilePath = rootFilePath + "/timeDependent/t.bin"
EFilePath = rootFilePath + "/timeDependent/E.bin"
dipoleAccelerationFilePath = rootFilePath + "/timeDependent/observables/dipoleAcceleration.bin"
normFilePath = rootFilePath + "/timeDependent/observables/norm.bin"

# # Potential and Hamiltonian

# # # Potential

potentialFilePath = rootFilePath + "/groundState/potential.bin"

# # # Real-Time Hamiltonian

hamiltonianTFilePath = rootFilePath + "/groundState/hamiltonianT.bin"
hamiltonianVFilePath = rootFilePath + "/groundState/hamiltonianV.bin"

# # # Imaginary-Time Hamiltonian

hamiltonianITFilePath = rootFilePath + "/groundState/hamiltonianiT.bin"
hamiltonianIVFilePath = rootFilePath + "/groundState/hamiltonianiV.bin"

# # # Wave Function

psiXFilePath = rootFilePath + "/groundState/psi_x.bin"
psiKFilePath = rootFilePath + "/groundState/psi_k.bin"

# Load Data

# # Position and Momentum Vectors

x = np.fromfile(xFilePath, dtype=float)
k = np.fromfile(kFilePath, dtype=float)

# # Absorbing Boundary Condition Vector

abc = np.fromfile(abcFilePath, dtype=float)

# # Time-Dependent Vectors

if (TD_SIMULATION == 1):

	t = np.fromfile(tFilePath, dtype=float)
	E = np.fromfile(EFilePath, dtype=float)
	dAcc = np.fromfile(dipoleAccelerationFilePath, dtype=complex)
	norm = np.fromfile(normFilePath, dtype=float)

# # Potential and Hamiltonian

# # # Potential

potential = np.fromfile(potentialFilePath, dtype=float)
potential = potential.reshape((N_x, N_x), order="C")

# # Hamiltonian

# # # Real-Time Hamiltonian

hamiltonianT = np.fromfile(hamiltonianTFilePath, dtype=complex)
hamiltonianT = hamiltonianT.reshape((N_x, N_x), order="C")

hamiltonianV = np.fromfile(hamiltonianVFilePath, dtype=complex)
hamiltonianV = hamiltonianV.reshape((N_x, N_x), order="C")

# # # Imaginary Time Hamiltonian

hamiltonianiT = np.fromfile(hamiltonianITFilePath, dtype=float)
hamiltonianiT = hamiltonianiT.reshape((N_x, N_x), order="C")

hamiltonianiV = np.fromfile(hamiltonianIVFilePath, dtype=float)
hamiltonianiV = hamiltonianiV.reshape((N_x, N_x), order="C")

# # Wave Function

psiX = np.fromfile(psiXFilePath, dtype=complex)
psiX = psiX.reshape((N_x, N_x), order="C")

psiK = np.fromfile(psiKFilePath, dtype=complex)
psiK = psiK.reshape((N_x, N_x), order="C")

# Generate Plots

# # Position and Momentum Vectors

linearPlot (x, x, path="../figures/parameters/x.png", xlabel=r'$x$ (a.u.)', ylabel=r'$x$ (a.u.)')
linearPlot (k, k, path="../figures/parameters/k.png", xlabel=r'$k$ (a.u.)', ylabel=r'$k$ (a.u.)')

# # Absorbing Boundary Vector

linearPlot (x, abc, path="../figures/parameters/abc.png", xlabel=r'$x$ (a.u.)', ylabel=r'Complex Potential $v_{im}(x)$ (a.u.)')

# # Time-Dependent Plots

if (TD_SIMULATION == 1):

	t = t / attosecond_au / 1000.0

	dAcc = applyFourierTransformWindow (dAcc)

	linearPlot (t, E, path="../figures/timeDependent/E.png", xlabel="Time (fs)", ylabel=r'$E(t)$')
	linearPlot (t, np.real(dAcc), path="../figures/timeDependent/dipoleAcceleration.png", xlabel="Time (fs)", ylabel=r'$\ddot{d}(t)$')
	linearPlot (t, norm, ymin=0.0, ymax=1.1, path="../figures/timeDependent/norm.png", xlabel="Time (fs)", ylabel=r'$n(t)$')

	dAcc = padFunction (dAcc)
	N_new = np.size(dAcc)

	t = np.linspace(- 0.5 * N_new, 0.5 * N_new - 1, N_new) * dt / attosecond_au / 1000.0

	w = np.linspace(- np.pi / dt, np.pi / dt, N_new) / electronvolt_au

	sacc = np.fft.fftshift(np.fft.fft(np.fft.fftshift(dAcc)))

	phase = np.unwrap(np.angle(sacc))

	linearPlot (w, np.power(np.absolute(sacc), 2.0), path="../figures/timeDependent/intensitySpectrum.png", xlabel=r'$\omega$ (eV)', ylabel=r'$| \ddot{d}(t) |^2$ (a.u.)', xmin=0.0, xmax=200.0, log=True)

	linearPlot (w, phase, path="../figures/timeDependent/spectralPhase.png", xlabel=r'$\omega$ (eV)', ylabel=r'$Phase (rad)', xmin=0.0, xmax=100.0, log=False, ymin=np.min(phase[np.absolute(w) < 100.0]), ymax=np.max(phase[np.absolute(w) < 100.0]))

X, Y = np.meshgrid(x, x)

# # Potential and Hamiltonian

# # # Potential

contourPlot (potential, X, Y, path="../figures/groundState/potential/potential.png", xlabel=r'$x_1$ (a.u.)', ylabel=r'$x_2$ (a.u.)', xMindx=(N_x // 2 - N_x // 8), xMaxdx=(N_x // 2 + N_x // 8), figsize=(8, (12.0 / 14.0) * 8))

if GENERATE_HAMILTONIAN_FIGURE:

	# # # Real-Time Hamiltonian

	contourPlot (np.real(hamiltonianT), X, Y, path="../figures/groundState/hamiltonian/realTime/HT_r.png", xlabel=r'$k_1$ (a.u.)', ylabel=r'$k_2$ (a.u.)')
	contourPlot (np.imag(hamiltonianT), X, Y, path="../figures/groundState/hamiltonian/realTime/HT_i.png", xlabel=r'$k_1$ (a.u.)', ylabel=r'$k_2$ (a.u.)')

	contourPlot (np.real(hamiltonianV), X, Y, path="../figures/groundState/hamiltonian/realTime/HV_r.png", xlabel=r'$x_1$ (a.u.)', ylabel=r'$x_2$ (a.u.)')
	contourPlot (np.imag(hamiltonianV), X, Y, path="../figures/groundState/hamiltonian/realTime/HV_i.png", xlabel=r'$x_1$ (a.u.)', ylabel=r'$x_2$ (a.u.)')

	# # # Imaginary-Time Hamiltonian

	contourPlot (hamiltonianiT, X, Y, path="../figures/groundState/hamiltonian/imagTime/HiT.png", xlabel=r'$k_1$ (a.u.)', ylabel=r'$k_2$ (a.u.)')
	contourPlot (hamiltonianiV, X, Y, path="../figures/groundState/hamiltonian/imagTime/HiV.png", xlabel=r'$x_1$ (a.u.)', ylabel=r'$x_2$ (a.u.)')

# # Wave Function

# # # Position Space

contourPlot (np.real(psiX), X, Y, path="../figures/groundState/waveFunction/psi_x_r.png", xlabel=r'$x_1$ (a.u.)', ylabel=r'$x_2$ (a.u.)', xMindx=(N_x // 2 - N_x // 8), xMaxdx=(N_x // 2 + N_x // 8), figsize=(8, (12.0 / 14.0) * 8))
# contourPlot (np.imag(psiX), X, Y, path="../figures/groundState/waveFunction/psi_x_i.png", xlabel=r'$x_1$ (a.u.)', ylabel=r'$x_2$ (a.u.)')

# # # Momentum Space

contourPlot (np.real(psiK), X, Y, path="../figures/groundState/waveFunction/psi_k_r.png", xlabel=r'$k_1$ (a.u.)', ylabel=r'$k_2$ (a.u.)')
# contourPlot (np.imag(psiK), X, Y, path="../figures/groundState/waveFunction/psi_k_i.png", xlabel=r'$k_1$ (a.u.)', ylabel=r'$k_2$ (a.u.)')
