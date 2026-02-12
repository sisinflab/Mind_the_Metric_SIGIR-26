def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break


from .gfcf import GFCF
from .svd_ae import SVD_AE
from .bspm import BSPM
from .chebycf import ChebyCF
from .pgsp import PGSP
from .fagsp import FaGSP
from .higsp import HiGSP

import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .lightgcn import LightGCN
        from .ultragcn import UltraGCN
        from .svd_gcn import SVDGCN
        from .svd_gcn_s import SVDGCNS
        from .gde import GDE
        from .sgde import SGDE
        from .rsgde import RSGDE
        from .csgde import CSGDE
        from .turbocf import TurboCF
        from .sgfcf import SGFCF