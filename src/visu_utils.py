from dataclasses import dataclass
from typing import ClassVar
from scipy.special import comb
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

    For 0<=i<=n and 0<=j<=m,
    C(i,j,n,m) = (n+m)! / i!j!(n+m-i-j)!.

    Parameters
    ----------
    n : int
        Number of lines of the output
    m : int
        Number of columns of the output
    """
    out = np.zeros((n,m), dtype=int)
    k2 = comb(n + m - 2, range(0,m)).astype(int)
    for i in range(0, m):
        out[:,i] = comb(n + m - 2 - i, range(0,n)).astype(int)
    for i in range(0, n):
        out[i] *= k2
    
    return out
