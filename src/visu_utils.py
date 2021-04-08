from dataclasses import dataclass
from typing import ClassVar
from scipy.special import comb, factorial
import numpy as np

@dataclass
class Verbose:
    """Utility class."""

    boolean: ClassVar[bool] = False
    
    @classmethod
    def classInit(cls, b):
        """Initialize the boolean class variable."""

        cls.boolean = b

    @classmethod
    def verboseprint(cls, *a, **k):
        """Prints if the class boolean variable is set to True."""
        
        if cls.boolean:
            print(*a,**k)
        else:
            lambda *a, **k: None

def comb2D(n, m):
    """
    Compute the 2D binomial coefficients.

    For 0<=i<n and 0<=j<m,
    C(i,j,n,m) = (n+m)! / i!j!(n+m-i-j)!.

    Parameters
    ----------
    n : int
        Number of lines of the output
    m : int
        Number of columns of the output
    """
    k1 = np.empty((n,m), dtype=float)
    for i in range(0, m):
        k1[:,i] = comb(n + m - i, range(0,n))
    k2 = comb(n + m, range(0,m))
    out = np.multiply(k1,k2)
    
    return out

def factorial2D(n,m):
    """
    Compute the 2D factorials.

    For 0<=i<n and 0<=j<m,
    fact(i,j) = i!j!.

    Parameters
    ----------
    n : int
        Number of lines of the output
    m : int
        Number of columns of the output
    """
    x = np.asmatrix(factorial(np.arange(n))).transpose()
    y = np.asmatrix(factorial(np.arange(m)))
    return np.dot(x, y)
