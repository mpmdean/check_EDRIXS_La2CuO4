import numpy as np
import scipy as sp
from scipy.special import voigt_profile
from matplotlib.ticker import AutoMinorLocator
from scipy.special import erf

import edrixs

    
def formatax(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(direction='in', which='both',
                  bottom=True, top=True, left=True, right=True)


B = 1e-3*np.array([1, 1, 0])
cf_splitting = 1e3
zeeman = sum(s*b for s, b in zip(edrixs.get_spin_momentum(2), B))
dd_levels = np.array([energy for dd_level in cf_splitting*np.arange(5)
                      for energy in [dd_level]*2], dtype=complex)
emat_rhb = np.diag(dd_levels)
emat = edrixs.cb_op(emat_rhb, edrixs.tmat_r2c('d', ispin=True)) + zeeman
_, eigenvectors = np.linalg.eigh(emat)


def get_eigenvector(orbital_index, spin_index):
    if orbital_index not in range(5) or spin_index not in range(2):
        raise Exception("Check orbital_index and spin_index")
    return eigenvectors[:, 2*orbital_index + spin_index]


D_Tmat = edrixs.get_trans_oper('dp32')


def get_F(vector_i, vector_f):
    """Scattering matrix
    Note that this is in hole notation."""
    F = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            F[i, j] = np.dot(np.dot(np.conj(vector_f.T), D_Tmat[i]),
                             np.dot(np.conj(D_Tmat[j].T), vector_i))
    return F


def peakshape(x, intensity, center, width, res):
    """Shape of a specific peak with lorentzin intrinsic width convolved with
    gaussian resolution function"""
    return intensity*voigt_profile(x - center, res/(2*np.sqrt(2*np.log(2))), width/2)


def one_transition_spectrum(x, energy, width,
                            orbital_index_i, spin_index_i,
                            orbital_index_f, spin_index_f,
                            thin, thout, phi, alpha,
                            res):
    """RIXS spectrum for a one paricular transition"""
    
    vector_i = get_eigenvector(orbital_index_i, spin_index_i)
    vector_f = get_eigenvector(orbital_index_f, spin_index_f)
    F = get_F(vector_i, vector_f)
    intensity = get_intensity(F, thin, thout, phi, alpha)
    rixs = peakshape(x, intensity, energy, width, res)
    
    # print(F.round(3))
    # print("\n\n")
    return rixs


def get_intensity(F, thin, thout, phi, alpha):
    """Total amplitude squared for one transition"""
    intensity = 0
    for beta in [0, np.pi/2]: # sum outgoing polarizations
        # print(f"thin={thin}\nthout={thout}\nalpha={alpha}\nbeta={beta}\n")
        ei, ef = edrixs.dipole_polvec_rixs(thin*np.pi/180, thout*np.pi/180,
                                           phi=phi*np.pi/180,
                                           alpha=alpha, beta=beta)
        intensity += np.abs(np.dot(ef, np.dot(F, ei)))**2
    return intensity


# Orbital energies (eV)
Ez2 = 1.7
Exy = 1.8
Ezx = 2.12

# Orbital widths (eV)
Wz2 = .14
Wxy = 0.1
Wx2y2 = 0.05
Wzx = 0.14

J = .13 # superexchange

res = .12 # Gaussian resolution FWHM

# Angles
alpha = 0*np.pi/2 # rad
thin = 119        # deg
thout = 11        # deg
phi = 0           # deg


def get_RIXS(x,
             Ez2=Ez2, Ezx=Ezx, Exy=Exy,
             Wz2=Wz2, Wzx=Wzx, Wxy=Wxy,
             J=J, res=res,
             I0=1,
             thin=thin, thout=thout, phi=phi,
             alpha=alpha,
             select=range(5)):
    
    # Ground state configuration
    orbital_index_i = 3
    spin_index_i = 0
    
    #              d3z2-r2            dzx       dzy   dx2-y2     dxy
    all_energies = [Ez2, Ez2+2*J/6, Ezx, Ezx, Ezx, Ezx, 0, 2*J, Exy, Exy]
    
    all_widths = [width for width_pair in [Wz2, Wzx, Wzx, Wx2y2, Wxy]
                  for width in [width_pair]*2]
    
    all_orbital_indices_f = [i for ipair in range(5) for i in [ipair]*2]
    
    all_spin_indices_f = [0, 1]*5
    
    I = sum(one_transition_spectrum(x, energy, width,
                                    orbital_index_i, spin_index_i,
                                    orbital_index_f, spin_index_f,
                                    thin, thout, phi, alpha,
                                    res)
            for energy, width, orbital_index_f, spin_index_f
            in zip(all_energies, all_widths, all_orbital_indices_f,
                   all_spin_indices_f)
            if orbital_index_f in select)
    return I0*I