import numpy as np

def rerange(img : np.ndarray, 
            min = 0.0, 
            max = 1.0):
  """Function that converts an image to the given desired range.
  
  Parameters
  ----------

  img : np.ndarray
    Image to be normalized.
  min : float
    Bottom value of the new range for conversion.
  max : float
    Top value of the new range for conversion."""
  return ((img - np.min(img))/(np.max(img) - np.min(img)))*(max - min) + min

def normalize(vec : np.ndarray) -> np.ndarray:
  """Normalizes an array or a set of arrays.

  Parameters
  ----------
  vec : np.ndarray (N, M)
    Array to be normalized."""
  return vec/np.tile(np.linalg.norm(vec, axis = 0), vec.shape[0]).reshape(vec.shape)


def rotation_matrix(alpha : float, 
                    beta : float, 
                    gamma : float):
    """Returns the final rotation matrix from the multiplication 
    of the three rotation matrices in the following order: Rxy*Rxz*Ryz, 
    which means it first rotates around the x axis, then around the y axis, 
    then around the z axis. 

    The matrix of rotations are:

    Rxy = np.matrix([[ np.cos(alpha), np.sin(alpha), 0.0], 
                     [-np.sin(alpha), np.cos(alpha), 0.0], 
                     [           0.0,           0.0, 1.0]])
    
    Rxz = np.matrix([[  np.cos(beta),           0.0,           np.sin(beta)], 
                     [         0.0,            1.0,                   0.0], 
                     [ -np.sin(beta),           0.0,           np.cos(beta)]])
    
    Ryz = np.matrix([[ 1.0,              0.0,             0.0], 
                     [ 0.0,    np.cos(gamma),   np.sin(gamma)], 
                     [ 0.0,   -np.sin(gamma),   np.cos(gamma)]])
    
    Parameters
    ----------
    alpha : float
      angle of rotation around the z axis.
    beta : float
      angle of rotation around the y axis.
    gamma : float
      angle of rotation around the x axis.
      
    """

    
    R = np.matrix([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma) - 
                    np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma) + 
                    np.sin(alpha)*np.sin(gamma)], 
                   [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma) + 
                    np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma) - 
                    np.cos(alpha)*np.sin(gamma)], 
                   [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])
    
    return R

def rotate(vec : np.ndarray, 
           alpha : float, 
           beta : float, 
           gamma : float):
    """Rotates a 3D array or set of arrays by a certain set of angles, in radians. 
    The order of rotation is Rxy*Rxz*Ryz, so it first rotates around 
    the x axis, then around the y axis, then around the z axis. 
    
    Parameters
    ----------
    vec : np.ndarray (N, 3)
      Array to be rotated.
    alpha : float
      angle of rotation around the z axis.
    beta : float
      angle of rotation around the y axis.
    gamma : float
      angle of rotation around the x axis.
      
    """
    
    R = rotation_matrix(alpha, beta, gamma)
    
    return np.asarray((R*np.matrix(vec).T).T)

def apply_B(M : np.ndarray, 
            B : np.ndarray, 
            gamma : float, 
            dt : float):
  """Applies a magnetic field B onto a magnetic moment M in a 
     certain interval dt, making it rotate about an angle w*dt, 
     with w = gamma*B_z, gamma being the gyromagnetic ratio of 
     the nucleus, and B_z the component of the magnetic field 
     orthogonal to the magnetic moment.
     
     Parameters
     ----------
     
     M : np.ndarray
      The magnetic moment of the atom
     B : np.ndarray
      The magnetic field to be applied.
     gamma : float
      The gyromagnetic moment of the nucleus
     dt : float
      The value of the simulation's dt"""

  x = normalize(M)
  y = np.cross(x, normalize(B))
  z = np.cross(x, y)

  B_z = np.dot(B, z)

  w = gamma*B_z
  f = w/(2.0*np.pi)

  angle = f*dt

  #first, a new coordinate system is defined 
  #so the rotations can be done in terms of angle
  new_coord = np.matrix([x,
                         y,
                         z]).T
  
  #it is also important to understand how do the original coordinate
  #system changes in the new one, so we can use it to come back
  
  #the vector is translated into the new coordinate system
  new_vec = np.array(new_coord*(np.matrix(M).T)).T[0]

  #the rotation is done
  rotated = rotate(new_vec, -angle, 0.0, 0.0)

  #we come back to the old coordinate system now with a rotated vector
  #by multiplying the vector with the inverse matrix of the transformation,
  #which is only the transpose matrix due to its orthonormality
  back = np.array((new_coord.T*np.matrix(rotated).T)).T[0]
  
  return back

def RMSE(img1, img2):
  """Function that calculates the [Root Mean Squared Error (RMSE)](https://en.wikipedia.org/wiki/Root_mean_square_deviation) between img1 and img2.
  
  Parameters
  ----------
  
  img1 : np.ndarray
    First image to be compared.
  img2 : np.ndarray
    Second image to be compared."""
  return np.sqrt((np.sum((img1 - img2)**2))/(float(img1.shape[0]*img1.shape[1])))

def gaussian_filter(size, epsilon):
    x_l, y_l = size
    x_ = np.arange(-x_l//2, x_l//2, 1.0)
    y_ = np.arange(-y_l//2, y_l//2, 1.0)
    x, y = np.meshgrid(x_, y_)
    
    g = (1/epsilon*np.sqrt(2*np.pi))*np.exp(-0.5*(x**2 + y**2)/epsilon**2)
    
    return rerange(g)


def sinc_f(x):
    """Returns a sinc function of x.
     
    Parameters
    ----------
    x : any
      Value(s)"""
    if type(x) == np.ndarray:
        f = np.sin(x)/x
        f[np.isnan(f)] = 1.0
        return f.astype(np.float32)
    elif x == 0.0:
        return 1.0
    else:
        return np.sin(x)/x
    
def square_f(x, e_square):
    """Returns a square function of x.
    
    Parameters
    ----------
    
    x : any
      Value(s) of x.
    e_square : float
      Square limit."""
    square = np.ones_like(x, dtype = np.float32)
    square[x < -e_square] = 0.0
    square[x > e_square] = 0.0
    return square