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
  augment.py    strong_augment                             CoTTA (also RoTTA's / BOSA's student input)
  align.py      AlignedBN2d (adaptive alpha, W2 constraint) BOSA (statistics alignment)
  sparsify.py   saliency_split, sparse_bn                  BOSA (activation sparsification)
  mixbn.py      MixBN2d (source-mixed tracked statistics)  MixTTA
  lame.py       lame (Laplacian output correction)         LAME (Boudiaf et al., 2022)
  common.py     forward, set_trainable, constants          shared

  baselines.py  Source, NormOnly
  rotta.py      RoTTA  (combines the RoTTA components)
  cotta.py      CoTTA  (combines the CoTTA components)
  bosa.py       BOSA   (combines the BOSA components)
  mixtta.py     MixTTA (forward-only: mixed statistics + flip/shift averaging + LAME)
"""
from .baselines import NormOnly, Source
from .bosa import BOSA
from .cotta import CoTTA
from .mixtta import MixTTA
from .rotta import RoTTA

__all__ = ["Source", "NormOnly", "RoTTA", "CoTTA", "BOSA", "MixTTA"]
