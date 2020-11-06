from dataclasses import dataclass
from typing import ClassVar

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