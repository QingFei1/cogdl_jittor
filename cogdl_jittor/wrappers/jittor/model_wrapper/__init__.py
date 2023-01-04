

from .base_model_wrapper import ModelWrapper, EmbeddingModelWrapper, UnsupervisedModelWrapper

import importlib

def register_model_wrapper(name):
    """
    New data wrapper types can be added to cogdl with the :func:`register_model_wrapper`
    function decorator.

    Args:
        name (str): the name of the model_wrapper
    """

    def register_model_wrapper_cls(cls):
        print("The `register_model_wrapper` API is deprecated!")
        return cls

    return register_model_wrapper_cls


def fetch_model_wrapper(name):
    if isinstance(name, type):
        return name
    if name in SUPPORTED_MW:
        path = ".".join(SUPPORTED_MW[name].split(".")[:-1])
        module = importlib.import_module(path)
    else:
        raise NotImplementedError(f"Failed to import {name} ModelWrapper.")
    class_name = SUPPORTED_MW[name].split(".")[-1]
    return getattr(module, class_name)


SUPPORTED_MW = {
    "node_classification_mw": "cogdl_jittor.wrappers.jittor.model_wrapper.node_classification.NodeClfModelWrapper",
    "grand_mw": "cogdl_jittor.wrappers.jittor.model_wrapper.node_classification.GrandModelWrapper",
    "dgi_mw": "cogdl_jittor.wrappers.jittor.model_wrapper.node_classification.DGIModelWrapper",
    "graphsage_mw": "cogdl_jittor.wrappers.jittor.model_wrapper.node_classification.GraphSAGEModelWrapper",
    "mvgrl_mw": "cogdl_jittor.wrappers.jittor.model_wrapper.node_classification.MVGRLModelWrapper",
}
