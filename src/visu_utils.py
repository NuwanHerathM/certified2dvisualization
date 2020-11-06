from dataclasses import dataclass, field
from typing import ClassVar, Optional

@dataclass
class Verbose:
    boolean: ClassVar[bool] = False
    
    @classmethod
    def classInit(cls, b):
        cls.boolean = b

    @classmethod
    def verboseprint(cls, *a, **k):
        if cls.boolean:
            print(*a,**k)
        else:
            lambda *a, **k: None