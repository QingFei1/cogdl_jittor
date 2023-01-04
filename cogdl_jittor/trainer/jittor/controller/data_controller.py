import torch
from cogdl_jittor.data import Dataset

from cogdl_jittor.wrappers.jittor.data_wrapper.base_data_wrapper import OnLoadingWrapper


class DataController(object):
    def __init__(self, world_size: int = 1, distributed: bool = False):
        self.world_size = world_size
        self.distributed = distributed

    # def distributed_dataloader(self, dataloader: DataLoader, dataset, rank):
    #     # TODO: just a toy implementation
    #     assert isinstance(dataloader, DataLoader)

    #     args, kwargs = dataloader.get_parameters()
    #     sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)
    #     kwargs["sampler"] = sampler
    #     dataloader = dataloader.__class__(*args, **kwargs)
    #     return dataloader

    def prepare_data_wrapper(self, dataset_w):
        
        dataset_w.pre_transform()
        dataset_w.prepare_training_data()
        dataset_w.prepare_val_data()
        dataset_w.prepare_test_data()
        return dataset_w

    def training_proc_per_stage(self, dataset_w):
        if dataset_w.__refresh_per_epoch__(): 
            dataset_w.prepare_training_data()
        return dataset_w
