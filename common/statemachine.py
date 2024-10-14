from enum import Enum

CONFIG_ENABLE_TIME_DELAY = False


class ObjectStateDef(str, Enum):
    UNKNOWN = "unknown"  # initial state
    FAILED = "failed"  # object not on the table
    SUCCESS = "success"  # folded successfully
    UNREACHABLE = "unreachable"  # unreachable state
    CRUMPLED = "crumpled"
    NEED_DRAG = "need_drag"
    SMOOTHED = "smoothed"
    FOLDED_ONCE = "folded_once"
    FOLDED_TWICE = "folded_twice"
    FOLDED_THIRD = "folded_third"  # halt


class RealDataStateDef(str, Enum):
    START = "start"
    INIT = "init"
    BEGIN_INSTANCE = "begin_instance"
    BEGIN_TRIAL = "begin_trial"
    TRIAL = "trial"
    END_TRIAL = "end_trial"
    END_INSTANCE = "end_instance"
    FINALIZE = "finalize"
    END = "end"


class ExperimentStateDef(str, Enum):
    START = "start"
    INIT = "init"
    BEGIN_EPISODE = "begin_episode"
    EPISODE = "episode"
    END_EPISODE = "end_episode"
    FINALIZE = "finalize"
    END = "end"
