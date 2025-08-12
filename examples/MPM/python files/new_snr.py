# %%
# from commpy.channels import awgn
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
# import pyMRI.pymri as pymri
# from pyMRI.utils import RMSE

# %% [markdown]
# ## Parâmetros

# %%

# Valores dos parâmetros dos metabólitos
METABOLITES = {"gaba"                   : (1.9346,   19.9*1e-3, 0.2917), # (δ, T2, A) [ppm, s, T]
               "naa"                    : (2.0050,   73.5*1e-3, 0.4289),
               "naag"                   : (2.1107,    6.6*1e-3, 0.0290),
               "glx2"                   : (2.1157,   90.9*1e-3, 0.0184),
               "gaba2"                  : (2.2797,   83.3*1e-3, 0.0451),
               "glu"                    : (2.3547,  116.3*1e-3, 0.0427),
               "cr"                     : (3.0360,   92.6*1e-3, 0.2026),
               "cho"                    : (3.2200,  113.6*1e-3, 0.0776),
               "m-ins3"                 : (3.2570,  105.3*1e-3, 0.0202),
               "m-ins"                  : (3.5721,  147.1*1e-3, 0.0411),
               "m-ins2"                 : (3.6461,  222.2*1e-3, 0.0150),
               "glx"                    : (3.7862,   45.7*1e-3, 0.1054),
               "cr2"                    : (3.9512,   40.0*1e-3, 0.2991),
               "cho+m-ins"              : (4.1233,    8.8*1e-3, 0.8244)}

# %% [markdown]
# ## Funções

# %%
hz_to_rad = lambda f : 2*np.pi*f # Converts hz to radians/s

def f_from_chem_shift(delta : float, 
                      B0 : float):
    """Returns the frequency in MHz (Megacycles/s) of a given compound 
    based on its chemical shift and magnetic field B0. It uses tetramethylsilane as reference.
    
    Parameters
    ----------
    
    delta : float                      [ppm]
        Chemical shift value, in ppm.
    B0 : float                         [T]
        Magnetic field B0, in T."""
    gamma = 42.577478461
    f_ref = gamma*B0
    return f_ref*(delta + 1.0)

def chem_shift_from_f(f : float, 
                      B0 : float):
    """Returns the chemical shift in ppm (parts per million) of a given compound 
    based on its frequency and magnetic field B0. It uses tetramethylsilane as reference.
    
    Parameters
    ----------
    
    f : float                       [MHz]
        Frequency value, in MHz.
    B0 : float                      [T]
        Magnetic field B0, in T."""
    gamma = 42.577478461
    f_ref =  gamma*B0
    return (f - f_ref)/f_ref

def setup_sim_t(t0 : float, tn : float, n_points : int, B0 : float, return_extra : bool = False):
    """Setups the simulation time parameters.
    
    Parameters
    ----------
    
    t0 : float  [s]
        Initial time.
    tn : float  [s]
        Final time.
    n_points : int 
        Number of points of the simulation.
    B0 : float  [T]
        Magnetic file.
    return_extra : bool = `False`
        Whether to return the time interval `Dt` and the sampling frequency `sampling_f`. Default is `False`."""
    Dt = tn - t0
    dt = (tn - t0)/n_points 
    sampling_f = 1.0/dt # cycles/s
    if return_extra == False:
        return t0, tn, dt, B0
    else:
        return t0, tn, dt, B0, Dt, sampling_f
    

def unpack_metabolites(metabolites : dict, B0 : float, met_slice : int = 15, return_deltas : bool = False):
    """Unpacks metabolite information from the METABOLITES variable. 
    
    Parameters
    ----------
    metabolites : dict
        METABOLITES information.
    B0 : float [T]
        Magnetic field for the frequencies calculation.
    met_slice : int = `15`
        Number of metabolites to account in the final array. Default is the maximum number of available metabolites, `15`.
    return_deltas : bool = `False`
        Whether to return the deltas instead of the frequencies of the metabolites. Default is `False`."""
    deltas = np.array( list(metabolites.values()) )[:met_slice, 0]
    t2s = np.array( list(metabolites.values()) )[:met_slice, 1]
    M_0s = np.array( list(metabolites.values()) )[:met_slice, 2]

    ws = hz_to_rad( f_from_chem_shift(deltas, B0) )
    if return_deltas == False:
        return ws, t2s, M_0s
    else:
        return deltas, t2s, M_0s
    
def population(ws : float, 
               T2s : np.ndarray, 
               M_0 : float, 
               phi : np.ndarray):
  """Returns a tuple with the population ordered parameters.
   
  Parameters
  ----------
  ws : float                                  [rad/s]
    Gamma of the population.
  T2s : np.ndarray                            [s]
    Decaying time T2.
  M_0 : float                                 [T]
    Initial magnetization value of each spin. 
  phi : np.ndarray                            [rad]
    Initial phase of each spin.
  """
  return (ws, T2s, M_0, phi)

def single_transverse_decay(t : np.ndarray, 
                            T2 : float,  
                            M_0: float, 
                            w : float, 
                            phi : float):
    """Simulates the transverse decay of the magnetization of a single spin.
    
    Parameters
    ----------
    t : np.ndarray (N, 1)                               [s]
      Time array of the simulation.
    T2 : float                                          [s]
      Decaying time of the transverse magnetization.
    M_0 : float                                         [T]
      Initial magnetization constant.
    w : float                                           [rad/s]
      Frequency of precession of the spin.
    phi : float                                         [rad]
      Phase of the magnetization.
    """
    return M_0*np.exp( 1j*(w*t + phi) )*np.exp(-t/T2)

def population_transverse_decay(t0 : float, 
                                tn : float, 
                                dt : float, 
                                population : tuple,  
                                echo : np.ndarray = np.array([]),
                                return_phase : bool = False):
  """Simulates the transverse decay of the magnetization of a population of spins. 
  Echoes before and after t0 and tn are filtered out.

  Parameters
  ----------
    t0 : float                                          [s]
      Initial time of the simulation.
    tn : float
      Final time of the simulation.                     [s]
    dt : float
      Time step of the simulation.                      [s]
    population : tuple (8)
      Tuple with all the information of the population, in order: 
      1. Angular frequency of the population.           [rad/s]
      2. Decaying time T2.                              [s]
      3. Initial magnetization value of each spin.      [T]
      4. Initial phase of each spin.                    [rad]
    echo : np.ndarray (M, 1)
      Echoes times to be applied to the simulation.
    return_phase : bool = `False`
      Whether to return the phases or not. Default is `False`."""

  w, T2, M_0, phi = population

  n = T2.shape[0]

  S = np.array([])

  if echo.size == 0:
     filtered_echoes = np.array([])
  else:
    filtered_echoes = echo[t0 < echo]
    filtered_echoes = filtered_echoes[filtered_echoes < tn]

  ts = np.concatenate((np.array([t0]), filtered_echoes, np.array([tn])))

  acc_phi = np.copy(phi)

  for k in range(ts.shape[0] - 1):
    t = np.arange(ts[k], ts[k + 1], dt)
    Dt = ts[k + 1] - t0
    theta = w*Dt

    S0 = np.sum(np.array([single_transverse_decay(t, T2[i], M_0[i], w[i], acc_phi[i]) for i in range(n)]), axis = 0)
    
    if ts[k + 1] != tn:
      acc_phi += - 2*(theta + acc_phi)

    S = np.concatenate((S, S0))

  acc_phi += 2*theta

  if return_phase == False:
    return S, np.arange(t0, tn, dt)
  else:
    return S, np.arange(t0, tn, dt), acc_phi
  

def fourier_spectrum(sig : np.ndarray, dt : float, B0 : float):
    """Returns the fourier spectrum and its frequencies, in terms of chemical shift, of a given signal.
    
    Parameters
    ----------
    sig : np.ndarray    [A.U.]
        Signal in which the fourier spectrum will be calculated.
    dt : float          [s]
        Time step.
    B0 : float          [T]
        Magnetic field for the chemical shift calculation."""
    sig_fft = np.fft.fftshift(np.fft.fft(sig, sig.size))
    freqs = chem_shift_from_f(np.fft.fftshift(np.fft.fftfreq(sig.size, d = dt)), B0)
    return freqs, sig_fft

def plot_chem_shifts(freqs : np.ndarray, 
                     sig_fft : np.ndarray, 
                     percentage : float, 
                     title : str = "Simulated MRS Spectra", 
                     xlabel : str = "δ (p.p.m.)",
                     ylabel : str = "Intensity (A.U.)",
                     c : str = "deeppink",
                     label : str = None,
                     plot_type : Literal["real", "imag", "abs"] = "abs",
                     linewidth = None): 
    """Plots a given spectrum in terms of its chemical shifts.
    
    Parameters
    ----------
    freqs : np.ndarray
      Frequencies array.
    sig_fft : np.ndarray
      Signal spectrum array.
    percentage : float (0.0, 1.0]
      Percentage of the signal to be displayed.
    title : str = `"Simulated MRS Spectra"`
      Title of the plot.
    xlabel : str = `"δ (p.p.m.)"`
      X-label of the plot.
    ylabel : str = `"Intensity (A.U.)"`
      y-label of the plot.
    c : str = `"deeppink"`
      Matplotlib color of the plot.
    label : str = `None`
      Label of the plot
    plot_type : Literal["real", "imag", "abs"] = `abs`
      Whether to plot the real, imaginary or absolute value of the array."""
    plot_freqs = freqs[freqs.size//2:]
    plot_sig_fft = sig_fft[sig_fft.size//2:]

    b = int(percentage*plot_freqs.size)

    _types = { "real" : np.real, "imag" : np.imag, "abs" : np.abs}
    
    plt.plot(plot_freqs[:b], _types[plot_type](plot_sig_fft)[:b], c = c, label = label, linewidth = linewidth)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if plt.gca().xaxis_inverted() == False:
      plt.gca().invert_xaxis() #inverts the x axis
    plt.grid(True)

# Definição de AWGN na biblioteca scikit-commpy
def awgn(input_signal, snr_dB, rate=1.0):
    """
    Addditive White Gaussian Noise (AWGN) Channel.

    Parameters
    ----------
    input_signal : 1D ndarray of floats
        Input signal to the channel.

    snr_dB : float
        Output SNR required in dB.

    rate : float
        Rate of the a FEC code used if any, otherwise 1.

    Returns
    -------
    output_signal : 1D ndarray of floats
        Output signal from the channel with the specified SNR.
    """

    avg_energy = sum(abs(input_signal) * abs(input_signal)) / len(input_signal)
    snr_linear = 10 ** (snr_dB / 10.0)
    noise_variance = avg_energy / (2 * rate * snr_linear)

    if isinstance(input_signal[0], complex):
        noise = (np.sqrt(noise_variance) * np.random.randn(len(input_signal))) + (np.sqrt(noise_variance) * np.random.randn(len(input_signal))*1j)
    else:
        noise = np.sqrt(2 * noise_variance) * np.random.randn(len(input_signal))

    output_signal = input_signal + noise

    return output_signal

# %% [markdown]
# ## Programa

# %%
metabolites = METABOLITES
met_i = 1

t0, tn, dt, B0 = setup_sim_t(0.0, 1.0, 2048, 3.0)

ws, t2s, M_0s = unpack_metabolites(metabolites, B0)
ws, t2s, M_0s = ws, t2s, M_0s

spins_phi = np.zeros_like(M_0s)

rcond = 1e-7 
zero_filtering = 1e-14

# %%
pop = population(ws, t2s, M_0s, spins_phi)

sig, t = population_transverse_decay( t0, tn, dt, pop, np.array([]) )
freqs, sig_fft = fourier_spectrum( sig, dt, B0 )

sig_fft = sig_fft/np.nanmax(sig_fft)

c_sig = awgn(sig_fft, 1.0)
# c_freqs, c_sig_fft = fourier_spectrum( c_sig, dt, B0 )

plt.figure(figsize=(20, 5))
plt.subplot(131)
plot_chem_shifts( freqs, awgn(sig_fft, 1.0), 1.0, label="SNR = 1 dB" )
plt.legend()

plt.subplot(132)
plot_chem_shifts( freqs, awgn(sig_fft, 10.0), 1.0, label="SNR = 10 dB" )
plt.legend()

plt.subplot(133)
plot_chem_shifts( freqs, awgn(sig_fft, 100.0), 1.0, label="SNR = 100 dB")
plt.legend()

plt.show()


