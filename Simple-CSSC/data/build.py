import os.path as osp
import torch

from .PRCC import PRCC
from torch.utils.data import DataLoader
from data_process.dataset_loader_cc import get_prcc_dataset_loader


__factory = {
    'prcc': PRCC,
}


def build_dataloader(cfg):
    """
    Build dataloader for PRCC dataset
    """
    if cfg.DATA.DATASET.lower() != 'prcc':
        raise ValueError("Only PRCC dataset is supported in this version")
    
    # Build datasets
    dataset = __factory[cfg.DATA.DATASET](dataset_root=cfg.DATA.ROOT)
    
    # Handle PRCC dataset
    class Args:
        def __init__(self):
            self.height = cfg.DATA.HEIGHT
            self.width = cfg.DATA.WIDTH
            self.train_batch = cfg.DATA.TRAIN_BATCH
            self.test_batch = cfg.DATA.TEST_BATCH
            self.num_instances = cfg.DATA.NUM_INSTANCES
            self.num_workers = cfg.DATA.NUM_WORKERS
            self.horizontal_flip_pro = cfg.AUG.RC_PROB
            self.pad_size = 10
            self.random_erasing_pro = cfg.AUG.RE_PROB
            self.dataset = 'prcc'
    
    args = Args()
    trainloader, query_sc_loader, query_cc_loader, galleryloader = get_prcc_dataset_loader(
        dataset, args, use_gpu=True
    )
    
    # For PRCC, queryloader is a list containing both query loaders
    queryloader = [query_sc_loader, query_cc_loader]
    num_classes = dataset.num_train_pids
    
    return trainloader, queryloader, galleryloader, num_classes
