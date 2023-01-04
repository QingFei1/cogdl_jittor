
# from cogdl_jittor.backend import BACKEND

# if BACKEND == 'jittor':
#     from .jittor import *
# elif BACKEND == 'torch':
#     from .torch import *
# else:
#     raise ("Unsupported backend:", BACKEND)




from cogdl_jittor.backend import BACKEND

if BACKEND == 'jittor':
    import jittor
    from .jittor import *
elif BACKEND == 'torch':
    import torch
    from .torch import *
else:
    raise ("Unsupported backend:", BACKEND)


import importlib
import inspect

from cogdl_jittor.data import Dataset
from cogdl_jittor.datasets import NodeDataset, GraphDataset, generate_random_graph


def register_dataset(name):
    """
    New dataset types can be added to cogdl with the :func:`register_dataset`
    function decorator.

    For example::

        @register_dataset('my_dataset')
        class MyDataset():
            (...)

    Args:
        name (str): the name of the dataset
    """

    def register_dataset_cls(cls):
        print("The `register_dataset` API is deprecated!")
        return cls

    return register_dataset_cls


def try_adding_dataset_args(dataset, parser):
    if dataset in SUPPORTED_DATASETS:
        path = ".".join(SUPPORTED_DATASETS[dataset].split(".")[:-1])
        module = importlib.import_module(path)
        class_name = SUPPORTED_DATASETS[dataset].split(".")[-1]
        dataset_class = getattr(module, class_name)
        if hasattr(dataset_class, "add_args"):
            dataset_class.add_args(parser)


def build_dataset_from_name(dataset, split=0):
    if isinstance(dataset, list):
        dataset = dataset[0]
    if isinstance(split, list):
        split = split[0]
    if dataset in SUPPORTED_DATASETS:
        path = ".".join(SUPPORTED_DATASETS[dataset].split(".")[:-1])
        module = importlib.import_module(path)
    else:
        dataset = build_dataset_from_path(dataset)
        if dataset is not None:
            return dataset
        raise NotImplementedError(f"Failed to import {dataset} dataset.")
    class_name = SUPPORTED_DATASETS[dataset].split(".")[-1]
    dataset_class = getattr(module, class_name)
    for key in inspect.signature(dataset_class.__init__).parameters.keys():
        if key == "split":
            return dataset_class(split=split)

    return dataset_class()


def build_dataset(args):
    if not hasattr(args, "split"):
        args.split = 0
    dataset = build_dataset_from_name(args.dataset, args.split)

    if hasattr(dataset, "num_classes") and dataset.num_classes > 0:
        args.num_classes = dataset.num_classes
    if hasattr(dataset, "num_features") and dataset.num_features > 0:
        args.num_features = dataset.num_features

    return dataset


def build_dataset_from_path(data_path, dataset=None):
    if dataset is not None and dataset in SUPPORTED_DATASETS:
        path = ".".join(SUPPORTED_DATASETS[dataset].split(".")[:-1])
        module = importlib.import_module(path)
        class_name = SUPPORTED_DATASETS[dataset].split(".")[-1]
        dataset_class = getattr(module, class_name)
        keys = inspect.signature(dataset_class.__init__).parameters.keys()
        if "data_path" in keys:
            dataset = dataset_class(data_path=data_path)
        elif "root" in keys:
            dataset = dataset_class(root=data_path)
        return dataset

    if dataset is None:
        try:
            if BACKEND == 'torch':
                return torch.load(data_path)
            elif BACKEND == 'jittor':
                return jittor.load(data_path)
        except Exception as e:
            print(e)
            exit(0)
    raise ValueError("You are expected to specify `dataset` and `data_path`")


SUPPORTED_DATASETS = {
    "cora": f"cogdl_jittor.datasets.{BACKEND}.planetoid_data.CoraDataset",
    "citeseer": f"cogdl_jittor.datasets.{BACKEND}.planetoid_data.CiteSeerDataset",
    "pubmed": f"cogdl_jittor.datasets.{BACKEND}.planetoid_data.PubMedDataset",
}
