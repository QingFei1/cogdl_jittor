from .base_data_wrapper import DataWrapper
import importlib


def register_data_wrapper(name):
    """
    New data wrapper types can be added to cogdl with the :func:`register_data_wrapper`
    function decorator.

    Args:
        name (str): the name of the data_wrapper
    """

    def register_data_wrapper_cls(cls):
        print("The `register_data_wrapper` API is deprecated!")
        return cls

    return register_data_wrapper_cls


def fetch_data_wrapper(name):
    if isinstance(name, type):
        return name
    if name in SUPPORTED_DW:
        path = ".".join(SUPPORTED_DW[name].split(".")[:-1])
        module = importlib.import_module(path)
    else:
        raise NotImplementedError(f"Failed to import {name} DataWrapper.")
    class_name = SUPPORTED_DW[name].split(".")[-1]
    return getattr(module, class_name)


SUPPORTED_DW = {
    "node_classification_dw": "cogdl_jittor.wrappers.jittor.data_wrapper.node_classification.FullBatchNodeClfDataWrapper",
    "graphsage_dw": "cogdl_jittor.wrappers.jittor.data_wrapper.node_classification.GraphSAGEDataWrapper",
}
