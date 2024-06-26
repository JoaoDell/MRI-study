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

def single_transverse_decay(t : np.ndarray, 
                            T2 : float,  
                            M_0: float, 
                            w : float, 
                            phi : float):
    """Simulates the transverse decay of the magnetization of a single spin.
    
    Parameters
    ----------
    t : np.ndarray (N, 1)
      Time array of the simulation.
    T2 : float
      Decaying time of the transverse magnetization.
    M_0 : float
      Initial magnetization constant.
    w : float
      Frequency of precession of the spin.
    phi : float
      Phase of the magnetization.
    """
    return M_0*np.exp( 1j*(w*t + phi) )*np.exp(-t/T2)

def B_gradient(x : np.ndarray, G : float, type : Literal["linear", "quadratic"]):
    """Generates a magnetic field gradient.

    Parameters
    ----------
    x : np.ndarray
      Positions.
    G : float
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
    
def generate_ws(B0 : float, pos : np.ndarray, G : float, gamma : float):
  """Generates the precession frequencies for the population.

  Parameters
  ----------
  B0 : float
    B0 field of the simulation.
  pos : np.ndarray
    Positions of the population.
  G : float
    Gradient value.
  gamma : float
    Gamma of the population.
  """
  DB = B_gradient(pos, G, 'linear')
  return gamma*(B0 + DB)

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
    t0 : float
      Initial time of the simulation.
    tn : float
      Final time of the simulation.
    dt : float
      Time step of the simulation.
    population : tuple (7)
      Tuple with all the information of the population, in order: 
      1. Spins positions.
      2. B0 value.
      3. Gradient value.
      4. Gamma of the population.
      5. Decaying time T2.
      6. Initial magnetization value of each spin. 
      7. Initial phase of each spin.
    echo : np.ndarray (M, 1)
      Echoes times to be applied to the simulation."""

  spins_x, B0, G, gamma, T2, M_0, phi = population
  w = generate_ws(B0, spins_x, G, gamma)

  n = T2.shape[0]

  S = np.array([])

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
    
def max_frequency(t0 : float, tn : float, dt : float):
  """Returns the maximum frequency that can be captured 
  by the given sampling parameters, according to the Nyquist rate.
  
  Parameters
  ----------
  t0 : float
    Initial time of the simulation.
  tn : float
    Final time of the simulation.
  dt : float
    Time step of the simulation."""
  return ((tn - t0)/dt)/2.0


def population(pos : np.ndarray, B0 : float, G : float, gamma : float, T2s : np.ndarray, M_0 : float, phi : np.ndarray):
  """Returns a tuple with the population ordered parameters.
   
  Parameters
  ----------
  pos : np.ndarray
    Spins positions.
  B0 : float
    B0 value.
  G : float
    Gradient value.
  gamma : float
    Gamma of the population.
  T2s : np.ndarray
    Decaying time T2.
  M_0 : float
    Initial magnetization value of each spin. 
  phi : np.ndarray
    Initial phase of each spin.
  """
  return (pos, B0, G, gamma, T2s, M_0, phi)

def event(t : float, echo : bool, ws : np.ndarray):
  """Returns a tuple with the ordered information of the event.
  
  Parameters
  ----------
  t : float
    Time of the event.
  echo : bool
    Whether an echo occurs or not in the time provided.
  ws : np.ndarray
    Frequencies of the spin population."""
  return (t, echo,  ws)