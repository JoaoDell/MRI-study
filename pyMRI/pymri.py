import numpy as np

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

def single_transverse_decay(t : np.ndarray, T2 : float,  M_0: float, w : float, phi : float):
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

def population_transverse_decay(t0 : float, tn : float, dt : float, T2 : np.ndarray, M_0: float, w : np.ndarray, phi : np.ndarray, echo : np.ndarray):
  """Simulates the transverse decay of the magnetization of a population of spins.

  Parameters
    ----------
    t0 : float
      Initial time of the simulation.
    tn : float
      Final time of the simulation.
    dt : float
      Time step of the simulation.
    T2 : np.ndarray (N, 1)
      Decaying times of the transverse magnetizations of each spin.
    M_0 : float
      Initial magnetization constant.
    w : np.ndarray (N, 1)
      Frequencies of precession of the spins.
    phi : float
      Phases of the magnetization of each spin.
    echo : np.ndarray (M, 1)
      Echoes times to be applied to the simulation."""
  n = T2.shape[0]

  S = np.array([])
   
  ts = np.concatenate((np.array([t0]), echo, np.array([tn])))

  acc_phi = np.copy(phi)

  for k in range(ts.shape[0] - 1):
    t = np.arange(ts[k], ts[k + 1], dt)
    Dt = ts[k + 1] - t0
    S0 = np.zeros_like(t)

    for i in range(n):
      S0 += single_transverse_decay(t, T2[i], M_0, w[i], acc_phi[i]).real
      theta = w[i]*Dt
      acc_phi[i] += np.pi - 2*theta - 2*acc_phi[i]
      
    S = np.concatenate((S, S0))

  return S, np.arange(t0, tn, dt) 
