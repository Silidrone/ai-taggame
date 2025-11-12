
from .direct_chase import DirectChasePolicy
from .intercept_chase import InterceptChasePolicy
from .corner_cut import CornerCutPolicy
from .zigzag_chase import ZigzagChasePolicy
from .spiral_chase import SpiralChasePolicy
from .random_walk import RandomWalkPolicy
from .ambush import AmbushPolicy
from .chaotic_chase import ChaoticChasePolicy
from .human_like import HumanLikePolicy

ALL_POLICIES = [
    DirectChasePolicy,
    InterceptChasePolicy,
    CornerCutPolicy,
    ZigzagChasePolicy,
    SpiralChasePolicy,
    RandomWalkPolicy,
    AmbushPolicy,
    ChaoticChasePolicy,
    HumanLikePolicy
]

