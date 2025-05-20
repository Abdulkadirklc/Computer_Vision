import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from configs.default import get_config
from data import build_dataloader
from models import build_model
from losses import build_losses
from evaluate.metrics import evaluate
from utils.util import AverageMeter, Logger, save_checkpoint, set_random_seed
from data.PRCC import PRCC
from data_process.dataset_loader_cc import get_prcc_dataset_loader
from train_cssc import train
from test_cc import test_for_prcc, get_data_for_cc
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



def parse_option():
    parser = argparse.ArgumentParser(description='Train image-based re-id model')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, help="market1501, cuhk03, dukemtmcreid, msmt17, prcc")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    # Training args for train_cssc.py
    parser.add_argument('--tri-start-epoch', type=int, default=5, help='epoch to start triplet loss')
    parser.add_argument('--print-train-info-epoch-freq', type=int, default=1, help='print training info every N epochs')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config, args


def main(config, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    use_gpu = torch.cuda.is_available()

    if not config.EVAL_MODE:
        sys.stdout = Logger(osp.join(config.OUTPUT, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(config.OUTPUT, 'log_test.txt'))
    print("==========\nConfig:{}\n==========".format(config))
    print("Currently using GPU {}".format(config.GPU))
    # Set random seed
    set_random_seed(config.SEED, use_gpu)

    # Build data loaders
    if config.DATA.DATASET.lower() == 'prcc':
        # Special handling for PRCC dataset
        class Args:
            def __init__(self):
                self.height = config.DATA.HEIGHT
                self.width = config.DATA.WIDTH
                self.train_batch = config.DATA.TRAIN_BATCH
                self.test_batch = config.DATA.TEST_BATCH
                self.num_instances = config.DATA.NUM_INSTANCES
                self.num_workers = config.DATA.NUM_WORKERS
                self.horizontal_flip_pro = config.AUG.RC_PROB
                self.pad_size = 10
                self.random_erasing_pro = config.AUG.RE_PROB
                self.dataset = 'prcc'
        
        loader_args = Args()
        dataset = PRCC(dataset_root=config.DATA.ROOT)
        trainloader, query_sc_loader, query_cc_loader, galleryloader = get_prcc_dataset_loader(
            dataset, loader_args, use_gpu=use_gpu
        )
        num_classes = dataset.num_train_pids
    else:
        # Regular dataset handling
        trainloader, queryloader, galleryloader, num_classes = build_dataloader(config)

    # Build model and classifier
    model, classifier = build_model(config, num_classes)
    
    # Create wrapper class to combine model and classifier for train_cssc.py
    class ModelWrapper(nn.Module):
        def __init__(self, base_model, classifier):
            super(ModelWrapper, self).__init__()
            self.base_model = base_model
            self.classifier = classifier
            
        def forward(self, x):
            features = self.base_model(x)
            outputs = self.classifier(features)
            return [features], [outputs]
    
    # Create a separate wrapper for testing that returns only features
    class FeatureExtractor(nn.Module):
        def __init__(self, base_model):
            super(FeatureExtractor, self).__init__()
            self.base_model = base_model
            
        def forward(self, x):
            return self.base_model(x)
    
    # Build classification and pairwise loss
    criterion_cla, criterion_pair = build_losses(config)
    
    # Build optimizer
    wrapper_model = ModelWrapper(model, classifier)
    parameters = list(wrapper_model.parameters())
    
    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                             weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))
    
    # Build lr_scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                        gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    start_epoch = config.TRAIN.START_EPOCH
    if config.MODEL.RESUME:
        print("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    feature_extractor = FeatureExtractor(model)
    
    if use_gpu:
        wrapper_model = nn.DataParallel(wrapper_model).cuda()
        feature_extractor = nn.DataParallel(feature_extractor).cuda()

    if config.EVAL_MODE:
        print("Evaluate only")
        if config.DATA.DATASET.lower() == 'prcc':
            test_for_prcc(args, query_sc_loader, query_cc_loader, galleryloader, feature_extractor, use_gpu)
        else:
            # Regular evaluation for other datasets
            pass
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = [-np.inf, -np.inf]  # For same clothes and changed clothes
    best_epoch = 0
    print("==> Start training")
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        start_train_time = time.time()
        
        # Use the train function from train_cssc.py
        train(args, epoch, trainloader, wrapper_model, optimizer, scheduler, 
              criterion_cla, criterion_pair, use_gpu)
              
        train_time += round(time.time() - start_train_time)        
        
        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
            (epoch+1) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:
            print("==> Test")
            
            if config.DATA.DATASET.lower() == 'prcc':
                rank1, mAP = test_for_prcc(args, query_sc_loader, query_cc_loader, galleryloader, 
                         feature_extractor, use_gpu, epoch=epoch+1)
                
                # Calculate average rank1 for both types
                avg_rank1 = sum(rank1) / len(rank1)
                avg_mAP = sum(mAP) / len(mAP)
                is_best = avg_rank1 > sum(best_rank1) / len(best_rank1)
                
                if is_best:
                    best_rank1 = rank1
                    best_epoch = epoch + 1

                    # Save model checkpoint
                    state_dict = model.state_dict()
                    save_checkpoint({
                        'state_dict': state_dict,
                        'rank1': rank1 if config.DATA.DATASET.lower() == 'prcc' else None,
                        'epoch': epoch,
                    }, is_best, osp.join(config.OUTPUT, 'checkpoint.pth.tar'))

    if config.DATA.DATASET.lower() == 'prcc':
        print("==> Best Same Clothes Rank-1 {:.1%}, Changed Clothes Rank-1 {:.1%}, achieved at epoch {}".format(
            best_rank1[0], best_rank1[1], best_epoch))
    else:
        print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1[0], best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


if __name__ == '__main__':
    config, args = parse_option()
    main(config, args)