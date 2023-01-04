
# from cogdl_jittor.backend import BACKEND

# if BACKEND == 'jittor':
#     from .jittor import *
# elif BACKEND == 'torch':
#     from .torch import *
# else:
#     raise ("Unsupported backend:", BACKEND)




import importlib
from cogdl_jittor.backend import BACKEND

if BACKEND == 'jittor':
    from .jittor import *
    from .jittor.base_model import BaseModel
elif BACKEND == 'torch':
    from .torch import *
else:
    raise ("Unsupported backend:", BACKEND)




def register_model(name):
    """
    New model types can be added to cogdl with the :func:`register_model`
    function decorator.
    For example::
        @register_model('gat')
        class GAT(BaseModel):
            (...)
    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        print("The `register_model` API is deprecated!")
        return cls

    return register_model_cls


def try_adding_model_args(model, parser):
    if model in SUPPORTED_MODELS:
        path = ".".join(SUPPORTED_MODELS[model].split(".")[:-1])
        module = importlib.import_module(path)
        class_name = SUPPORTED_MODELS[model].split(".")[-1]
        getattr(module, class_name).add_args(parser)


def build_model(args):
    model = args.model
    if isinstance(model, list):
        model = model[0]
    if model in SUPPORTED_MODELS:
        path = ".".join(SUPPORTED_MODELS[model].split(".")[:-1])
        module = importlib.import_module(path)
    else:
        raise NotImplementedError(f"Failed to import {model} model.")
    class_name = SUPPORTED_MODELS[model].split(".")[-1]
    return getattr(module, class_name).build_model_from_args(args)


SUPPORTED_MODELS = {
    "gcn": f"cogdl_jittor.models.{BACKEND}.nn.gcn.GCN",
    "gat": f"cogdl_jittor.models.{BACKEND}.nn.gat.GAT",
    "grand": f"cogdl_jittor.models.{BACKEND}.nn.grand.Grand",
    "gcnii": f"cogdl_jittor.models.{BACKEND}.nn.gcnii.GCNII",
    "dgi": f"cogdl_jittor.models.{BACKEND}.nn.dgi.DGIModel",
    "graphsage": f"cogdl_jittor.models.{BACKEND}.nn.graphsage.Graphsage",
    "drgat": f"cogdl_jittor.models.{BACKEND}.nn.drgat.DrGAT",
    "mvgrl": f"cogdl_jittor.models.{BACKEND}.nn.mvgrl.MVGRL",
}
