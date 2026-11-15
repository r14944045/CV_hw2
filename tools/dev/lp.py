import os, sys
import fe
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../starter"))
from tta import MixTTAMS
for spec in sys.argv[1:]:
    kw = eval(spec)
    fe.longproxy(lambda model, kw=kw: MixTTAMS(model, **kw), tag=spec)
