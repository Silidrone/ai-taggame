"""
Collection of deterministic chaser policies with different strategies.
Each policy is completely deterministic and uses a different principle.
"""

from .direct_chase import DirectChasePolicy
from .intercept_chase import InterceptChasePolicy
from .corner_cut import CornerCutPolicy
from .zigzag_chase import ZigzagChasePolicy
from .spiral_chase import SpiralChasePolicy
from .wall_hug import WallHugPolicy
from .center_control import CenterControlPolicy
from .random_walk import RandomWalkPolicy
from .patrol import PatrolPolicy
from .ambush import AmbushPolicy

ALL_POLICIES = [
    DirectChasePolicy,
    InterceptChasePolicy,
    CornerCutPolicy,
    ZigzagChasePolicy,
    SpiralChasePolicy,
    WallHugPolicy,
    CenterControlPolicy,
    RandomWalkPolicy,
    PatrolPolicy,
    AmbushPolicy
]
