import matplotlib.pyplot as plt
import numpy as np
from typing import Literal

T1_EXAMPLES = {'Substância Branca' : 790*1e-3, 
               'Substância cinzenta' : 920*1e-3, 
               'Líquido céfalo-raquidiano (líquor)' : 4000*1e-3,
               'Sangue (arterial)' : 1200*1e-3,
               'Parênquima hepático' : 490*1e-3,
               'Miocárdio' : 870*1e-3,
               'Músculo' : 870*1e-3,
               'Lipídios (gordura)' : 260*1e-3}

T2_EXAMPLES = {'Substância Branca' : 90*1e-3, 
               'Substância cinzenta' : 100*1e-3, 
               'Líquido céfalo-raquidiano (líquor)' : 2000*1e-3,
               'Sangue (arterial)' : 50*1e-3,
               'Parênquima hepático' : 40*1e-3,
               'Miocárdio' : 60*1e-3,
               'Músculo' : 50*1e-3,
               'Lipídios (gordura)' : 80*1e-3}

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

# The below was calculated in ..\examples\sigma_relations.ipynb
A = 0.4935147915609802
STD_A = 0.00234223322864983
C = 0.3059269851116157
STD_C = 0.0019239898238883272
D = 1.1590522266759649
STD_D = 0.00010226500977286893
E = -1.0029749238349337
STD_E = 0.032606022203370645

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

def B_gradient(x : np.ndarray, 
               G : float, 
               type : Literal["constant", "linear", "quadratic"]):
    """Generates a magnetic field gradient.

    Parameters
    ----------
    x : np.ndarray    [m]
      Positions.
    G : float         [T/m]
      Gradient rate
    type : str
      Gradient shape. The gradient shapes available are:
      1. "linear" : A linear gradient shape.
      2. "quadratic" : A quadratic gradient shape.
      """
    if type == "linear":
        return G*x
    elif type == "quadratic":
        return G*x**2
    elif type == "constant":
        return G*np.ones_like(x)
    
def generate_ws(B0 : float, 
                pos : np.ndarray, 
                G : float, 
                gamma : float, 
                gradient : Literal["constant", "linear", "quadratic"]):
  """Generates the precession frequencies for the population.

  Parameters
  ----------
  B0 : float                      [T]
    B0 field of the simulation.
  pos : np.ndarray                [m]
    Positions of the population.
  G : float                       [T/m]
    Gradient value.
  gamma : float                   [MHz/T]
    Gamma of the population.
  """
  DB = B_gradient(pos, G, gradient)
  return -gamma*(B0 + DB)

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
  
def corrupted_lw(T2 : np.ndarray,
                 s : float,
                 ws : float, 
                 M_0: float,
                 phis : float):
   """Returns a population with a range of T2s for linewidth (LW) broadening.
    
      Parameters
      ----------
      T2 : np.ndarray                                               
        Decaying time T2.                                      [s]
      s : float (0.0, +inf]
        Scalling constant for T2s.
      ws : float
        Angular frequencies of the population.           [rad/s]       
      M_0 : float        
        Initial magnetization value.                           [T]
      phis : float        
        Initial phase.                                         [rad]
        """
   T2s = s*T2
   return population(ws, T2s, M_0, phis)

def snr(signal : np.ndarray, percent : float = 0.1, plot : bool = False):
   """Calculates the signal-to-noise ratio of a given signal. 
      The SNR is calculated as the ratio of two values:
      1. The maximum value of the signal, denoted by S.
      2. The standard deviation of the signal at its farthest from the origin region.
      
      Parameters
      ----------

      signal : np.ndarray
        Signal to have the SNR calculated.
      percent : float
        The percentage of the signal size 
        that defines the size of the interval 
        from its end that will have the standard deviation calculated.
      """
   assert 0.01 <= percent <= 1.0, "Percent must be in the interval [0.01, 1.0]"
   
   n = signal.size
   r = int(percent*n)

   S = np.nanmax(np.abs(signal))
   N = np.std(signal[n - r:n])

   if plot == True:
      x = np.arange(n - r, n, 1.0)
      plt.plot(x, signal[n - r:n])
   
   return S/N 

def corrupted_snr(signal : np.ndarray,
                  center : float, 
                  sigma : float,  
                  a : float = 1.0,
                  offset : float = 0.0, 
                  add_sig : bool = True,
                  bitgenerator : np.random.BitGenerator = np.random.MT19937()):
   """Returns the given signal with a normal (gaussian) white noise addition of given parameters.
   
   Parameters
   ----------
   signal : np.ndarray
    Original signal to be corrupted.
   center : float
    White noise center.
   sigma : float
    Standard deviation of the noise.
   a : float, optional
    Amplitude of the white noise. Default is 1.0
   offset : float, optional
    White noise offset, default is 0.0.
   add_sig : bool, optional
    Original signal addition condition. Default is ``True``, so signal will be added.
   bitgenerator : np.random.BitGenerator, optional
    Numpy bitgenerator for the given noise generation. Default is MT19937."""
   gen = np.random.Generator(bitgenerator)
   n = signal.size
   noise = gen.normal(center, sigma, n) + 1j*gen.normal(center, sigma, n)
   return signal*add_sig + a*noise + offset
   
def max_frequency(dt : float):
  """Returns the maximum frequency, in Hz that can be captured 
  by the given sampling parameters, according to the Nyquist rate.
  
  Parameters
  ----------
  dt : float                        [s]
    Time step of the simulation."""
  return (1.0/dt)/2.0

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

def event(t : float, 
          echo : bool, 
          ws : np.ndarray):
  """Returns a tuple with the ordered information of the event.
  
  Parameters
  ----------
  t : float                                             [s]
    Time of the event.
  echo : bool                                            
    Whether an echo occurs or not in the time provided.
  ws : np.ndarray                                       [rad/s]
    Frequencies of the spin population."""
  return (t, echo,  ws)

hz_to_rad = lambda f : 2*np.pi*f # Converts hz to radians/s
rad_to_hz = lambda w : w/(2*np.pi) # Converts radians/s to hz

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

def check_frequency(w : float, 
                    dt : float,
                    print_checks : bool = False):
    """Checks if a given frequency or group of frequencies can be captured by a sampling step.
    
    Parameters
    ----------
    w : float                         [rad/s]
      Frequency or array of frequencies.
    dt : float                        [s]
      Time step of the simulation. 
    print_checks : bool = `False`
      Whether to print or no the checked array. If true, will print a boolean array. Default is set to `False`.
    """
    array = rad_to_hz(w) <= max_frequency(dt)
    if print_checks==True:
      print(array)
    return np.all(array)

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

def y2y1_matrices(  sig : np.ndarray, 
                    L : float, 
                    return_y : bool = False):
    """Generates the Y matrix for the MPM calculations.
    
    Parameters
    ----------
    sig : np.ndarray (N, 1)
        The signal from which the matrix will be generated.
    L : float [0.0, 1.0]
        The percentage of the size of the signal where the signal will be sliced.\n
        Signal slice will be of size `int(L*N)`, where `N` is the total size of the signal.
    return_y : bool = False
        Whether to output only the major Y matrix"""
    N = sig.shape[0]
    L_ = int(L*N)
    Y = np.zeros((N - L_, L_ + 1), dtype=np.complex128)
    for i in range(N - L_):
        Y[i] = sig[i:i + L_ + 1]
    if return_y == False:
        return Y[:,1:], Y[:,:L_] #Y2, Y1
    else:
        return Y

def filter_sig(sig : np.ndarray, 
               L : float, 
               noise_threshold : float, 
               rcond : float = 1e-7,
               zero_filtering : float = 1e-15,
               return_poles_and_res : bool = False,
               return_full_arrays : bool = False):
    """Filters a signal using the MPM algorithm. Returns the reconstructed and filtered signal as default, 
    but can return the poles and residues if `return_poles_and_res` is set to `True`.
    
    Parameters
    ----------
    sig : np.ndarray
        Signal to be filtered.
    L : float `[0.0, 1.0]`
        The percentage of the size of the signal where the signal will be sliced.\n
        Signal slice will be of size `int(L*N)`, where `N` is the total size of the signal.
    noise_threshold : float
        Threshold for filtering the singular values. 
        `singular_value/max_singular_value <= noise_threshold` will be filtered out.
    rcond : float = `1e-7`
        Threshold for filtering singular values in the Moore-Penrose and least-squares steps. 
        Default is set to `1e-7`.
    return_poles_and_res : bool = `False`
        Whether to return or not the poles and residues. If True, will return the reconstructed signal, 
        the poles and residues in a tuple.
    return_full_arrays : bool = `False`
      Whether to return the full arrays or sliced where the values are near-zero, delimited by the `noise_threshold` variable. 
      Default is `False`."""
    # Matrix Y generation step
    N = sig.size
    L_ = int(L*N)
    Y = y2y1_matrices(sig, L, return_y=True)

    # SVD noise filtering step
    U, Sigma, Vt = np.linalg.svd(Y, full_matrices=False)
    max_sval = Sigma[0]
    M = Sigma[Sigma/max_sval > noise_threshold].size
    # below an alternate step that considers the percentage of contribution 
    # of the total eigenvalues and not the singular values (singular values = sqrt(eigenvalues))
    # M = Sigma[Sigma**2/np.sum(Sigma**2) > noise_threshold].size 
    Y_ = np.matmul( np.matmul(U, np.diag(Sigma)[:, :M]), Vt[:M, :] ) # filtered Y

    Y2, Y1 = Y_[:,1:], Y_[:,:L_] #Y2, Y1

    # Matrix Y1^+Y2 construction step
    Y1_p = np.linalg.pinv(Y1, rcond=rcond)
    A = np.matmul(Y1_p, Y2)

    # Eigenvalues calculation step
    w = np.linalg.eigvals(A)

    # Zero values filtering
    w.real[np.abs(w.real) <= zero_filtering] = 0
    w.imag[np.abs(w.imag) <= zero_filtering] = 0
    if return_full_arrays == False:
      w = w[np.abs(w) > zero_filtering]

    # Residues calculation step
    Zs = np.zeros((N, w.shape[0]), dtype=np.complex128)
    for i in range(N):
        Zs[i] = np.power(w, i)

    R = np.linalg.lstsq(Zs, sig, rcond=rcond)[0].astype(np.complex128)

    # Zero values filtering
    R.real[np.abs(R.real) <= zero_filtering] = 0
    R.imag[np.abs(R.imag) <= zero_filtering] = 0
    if return_full_arrays == False:
      R = R[np.abs(R) > zero_filtering]

    L_f = R.size

    # Reconstruction of the filtered signal step
    reconstructed_sig = np.zeros(N, dtype=np.complex128)
    for i in range(L_f):
        reconstructed_sig += R[i]*Zs[:,i]
    
    if return_poles_and_res == False:
      return reconstructed_sig
    else: 
       return reconstructed_sig, w, R
  
def calculate_variables_from_z_and_r(z : np.complex128, r : np.complex128, ts : float):
    """Calculates the variables `S_0`, `phi`, `omega` and `alpha` from the poles `z` and residues `r`, 
    calculated from a signal using the MPM algorithm.
    
    Parameters
    ----------
    z : np.complex128
        Pole value or poles array.
    r : np.complex128
        Residue value or residues array.
    ts : float
        Sampling period.
        
    Returns
    -------
    out : tuple
      Returns a tuple with the calculated variables arrays in the following order: `s0`, `phi`, `omega`, `alpha`"""
    # a, b, c, d = z.real, z.imag, r.real, r.imag

    s0 = np.sqrt(r.real**2 + r.imag**2)
    phi = np.piecewise(r, [np.arctan2(r.imag, r.real) >= 0, 
                           np.arctan2(r.imag, r.real) < 0], 
                          [lambda x : np.arctan2(x.imag, x.real),
                           lambda x : np.arctan2(x.imag, x.real) + 2*np.pi]).real
    omega = (1/ts)*np.piecewise(z, [np.arctan2(z.imag, z.real) >= 0, 
                                    np.arctan2(z.imag, z.real) < 0], 
                                   [lambda x : np.arctan2(x.imag, x.real),
                                    lambda x : np.arctan2(x.imag, x.real) + 2*np.pi]).real
    alpha = np.piecewise(z, [np.logical_or(z.real != 0, z.imag != 0), 
                             np.logical_and(z.real == 0, z.imag == 0)], 
                            [lambda x :  (-1/ts)*np.log( np.sqrt( x.real**2 + x.imag**2 )),
                             lambda x : np.inf*(1 + 1j)]).real 
    # above it was supposed to be -inf but it messes with the zero so it makes no difference
    return s0, phi, omega, alpha

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
    

def snr_to_sigma(snr : float):
    """Returns a sigma value that will approximately corresponds to a given SNR value.
    
    Parameters
    ----------
    
    snr : float
        Desired SNR value.
    expected_peak : float
        The signal's expected peak."""
    actual_snr = (snr - E)/D # Linear correction (inverse of the linear function)
    given_sigma = 1.0/actual_snr

    b = np.log(A)
    return (b - np.log(A - given_sigma))/C # inverse of the model function

def select_peaks(s0 : np.ndarray,
                 phi : np.ndarray,
                 omega : np.ndarray,
                 t2 : np.ndarray, 
                 percentage : float = 0.05):
    """Calculates the peaks to be individually analysed, based on the highest peak value of `s0`. 
    Peaks below `percentage*highest_peak` will be discarded.
    
    Parameters
    ----------
    s0 : np.ndarray
        Peaks array.
    phi : np.ndarray
        Phases array.
    omega : np.ndarray
        Frequencies array.
    t2 : np.ndarray
        Decaying time T_2 array.
    percentage : float = `0.1`
        The percentage of the peak that will be used to filter. Default is `0.1`."""
    
    sort_arr = np.argsort(s0)

    s0_ = s0[sort_arr]
    phi_ = phi[sort_arr]
    omega_ = omega[sort_arr]
    t2_ = t2[sort_arr]

    peak = s0_[-1]

    s0__ = s0_[s0_ >= percentage*peak]
    phi__ = phi_[s0_ >= percentage*peak]
    omega__ = omega_[s0_ >= percentage*peak]
    t2__ = t2_[s0_ >= percentage*peak]

    n_comp = s0__.size

    return s0__, phi__, omega__, t2__, n_comp