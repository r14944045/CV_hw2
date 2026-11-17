"""Stage-wise BN-affine learning rates (r5/selftrain_stage_lr.py) combined with the depth-dependent source weight
(r5/erroranalysis_depth_mix.py)."""
from r5.erroranalysis_depth_mix import DepthMix
from r5.selftrain_stage_lr import SelfTrainStageLR


class StageLRDepthMix(SelfTrainStageLR, DepthMix):
    pass
