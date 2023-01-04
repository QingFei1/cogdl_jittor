
from cogdl_jittor.backend import BACKEND

if BACKEND == 'jittor':
    from .jittor.trainer import Trainer
    
elif BACKEND == 'torch':
    from .torch.trainer import Trainer  
else:
    raise ("Unsupported backend:", BACKEND)

