
from cogdl_jittor.backend import BACKEND

if BACKEND == 'jittor':
    from .jittor.data_wrapper import register_data_wrapper, fetch_data_wrapper
    from .jittor.model_wrapper import (
        register_model_wrapper,
        fetch_model_wrapper,
        ModelWrapper,
        EmbeddingModelWrapper,
    )
elif BACKEND == 'torch':
    from .torch.data_wrapper import register_data_wrapper, fetch_data_wrapper
    from .torch.model_wrapper import (
        register_model_wrapper,
        fetch_model_wrapper,
        ModelWrapper,
        EmbeddingModelWrapper,
    )
else:
    raise ("Unsupported backend:", BACKEND)

