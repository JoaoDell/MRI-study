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

METABOLITES = {"glutamato"              : (3.7444,  89*1e-3), # (δ, T2) [ppm, s]
               "glutamina"              : (3.7625, 116*1e-3),
               "gaba"                   : (3.0082, 105*1e-3),
               "myo-inositol"           : (3.5177, 148*1e-3),
               "phosphorylethanolamine" : (3.9825,  96*1e-3),
               "phosphocreatine"        : (3.0280, 113*1e-3),
               "taurina"                : (3.4190,  93*1e-3),
               "naa"                    : (2.0050, 202*1e-3)}

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

def B_gradient(x : np.ndarray, G : float, type : Literal["constant", "linear", "quadratic"]):
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
    
def generate_ws(B0 : float, pos : np.ndarray, G : float, gamma : float, gradient : Literal["constant", "linear", "quadratic"]):
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
                                echo : np.ndarray,
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
      Echoes times to be applied to the simulation."""

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

    S0 = np.sum(np.array([single_transverse_decay(t, T2[i], M_0, w[i], acc_phi[i]).real for i in range(n)]), axis = 0)
    
    if ts[k + 1] != tn:
      acc_phi += - 2*(theta + acc_phi)

    S = np.concatenate((S, S0))

  acc_phi += 2*theta

  if return_phase == False:
    return S, np.arange(t0, tn, dt)
  else:
    return S, np.arange(t0, tn, dt), acc_phi
  
def corrupted_lw(w : float, 
                 size : float, 
                 dw : float,
                 T2 : float,
                 M_0: float,
                 phi : float):
   """Returns a population with a range of T2s for linewidth (LW) broadening.
    
      Parameters
      ----------
      w : float
        Central angular frequency of the population.           [rad/s]
      size : float [0.0, +inf]                  
        Size, in percentage of w (%), of the range radius of w.
      dw : float                                               [s]
        w step.          
      T2 : float                                               [s]
        Decaying time T2.                                      [s]
      M_0 : float        
        Initial magnetization value.                           [T]
      phi : float        
        Initial phase.                                         [rad]
        """
   w_0 = w*( 1.0 - size )
   w_n = w*( 1.0 + size )

   ws = np.arange(w_0, w_n, dw)
   T2s = np.repeat(T2, ws.size)
   phis = np.repeat(phi, ws.size)

   return population(ws, T2s, M_0, phis)

def corrupted_snr():
   pass
    
def max_frequency(dt : float):
  """Returns the maximum frequency, in Hz that can be captured 
  by the given sampling parameters, according to the Nyquist rate.
  
  Parameters
  ----------
  dt : float                        [s]
    Time step of the simulation."""
  return (1.0/dt)/2.0

def population(ws : float, T2s : np.ndarray, M_0 : float, phi : np.ndarray):
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

def event(t : float, echo : bool, ws : np.ndarray):
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

def f_from_chem_shift(delta : float, B0 : float):
    """Returns de frequency in MHz (Megacycles/s) of a given compound 
    based on its chemical shift and magnetic field B0.
    
    Parameters
    ----------
    
    delta : float                       [ppm]
        Chemical shift value, in ppm.
    B0 : float                          [T]
        Magnetic field B0, in T."""
    gamma = 42.58
    f_ref = gamma*B0
    return f_ref*(delta*1e-6 + 1.0)

hz_to_rad = lambda f : 2*np.pi*f # Converts hz to radians/s
rad_to_hz = lambda w : w/(2*np.pi) # Converts radians/s to hz

def f_from_chem_shift(delta : float, B0 : float):
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

def chem_shift_from_f(f : float, B0 : float):
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

def check_frequency(w : float, dt : float):
  return np.all(rad_to_hz(w) <= max_frequency(dt))
