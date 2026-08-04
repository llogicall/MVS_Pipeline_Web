import torch
from torch.utils import data
import torch.distributed as dist
from torch.utils.data import RandomSampler, SequentialSampler
import importlib

from .dtu_yao import MVSDataset as DtuDataset
from .general_eval import MVSDataset as EvalDataset
from .blendedmvs import BlendedMVSDataset
from .aerial_mvs import MVSDataset as AerialMVSDataset


def find_dataset_def(dataset_name):
    """查找数据集定义，保持项目统一数据集接口。"""
    module_name = 'datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    return getattr(module, "MVSDataset")


def get_loader(args, datapath, listfile, nviews, mode="train"):
    if args.dataset_name == "dtu_yao":
        dataset = DtuDataset(datapath, listfile, mode, nviews, args.img_size, args.numdepth, args.interval_scale)
    elif args.dataset_name == "general_eval":
        dataset = EvalDataset(datapath, listfile, mode, nviews, args.numdepth, args.interval_scale, args.inverse_depth,
                              max_h=args.max_h, max_w=args.max_w, fix_res=args.fix_res)
    elif args.dataset_name == "blendedmvs":
        dataset = BlendedMVSDataset(datapath, listfile, mode, nviews, args.numdepth, args.interval_scale)
    elif args.dataset_name == "aerial_mvs":
        dataset = AerialMVSDataset(datapath, mode, nviews, args.img_size, args.numdepth, args.interval_scale)
        
    
    else:
        raise NotImplementedError("Don't support dataset: {}".format(args.dataset_name))

    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    else:
        sampler = RandomSampler(dataset) if (mode == "train") else SequentialSampler(dataset)

    data_loader = data.DataLoader(dataset, args.batch_size, sampler=sampler, num_workers=4, drop_last=(mode == "train"), pin_memory=True)

    return data_loader, sampler
