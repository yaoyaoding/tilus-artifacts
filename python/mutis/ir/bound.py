from typing import Optional


class Bound:
    def __init__(self, lower: Optional[int] = None, upper: Optional[int] = None):
        self.lower: Optional[int] = lower
        self.upper: Optional[int] = upper
