"""Modular test-time adaptation components.

  file          component                                  paper
  ------------  -----------------------------------------  ------------------------
  norm.py       RobustBN2d, configure_norm                 RoTTA (RBN); CoTTA/TENT (batch BN)
  memory.py     CSTU memory bank                           RoTTA
  losses.py     timeliness_weight                          RoTTA
                soft_cross_entropy, entropy                both
  teacher.py    MeanTeacher (EMA teacher)                  both (CoTTA: weight-averaged pseudo-labels)
                augmentation_averaged_logits,              CoTTA
                low_confidence (anchor gate)
  restore.py    StochasticRestore                          CoTTA
  augment.py    strong_augment                             CoTTA (also RoTTA's student input)
  common.py     forward, set_trainable, constants          shared

  baselines.py  Source, NormOnly
  rotta.py      RoTTA  (combines the RoTTA components)
  cotta.py      CoTTA  (combines the CoTTA components)
"""
from .baselines import NormOnly, Source
from .cotta import CoTTA
from .rotta import RoTTA

__all__ = ["Source", "NormOnly", "RoTTA", "CoTTA"]
