import importlib


def find_dataset_def(dataset_name):
    """查找数据集定义，优先从 shared/datasets/ 加载"""
    # 优先从当前项目的 shared/datasets/ 加载
    try:
        module = importlib.import_module('datasets.{}'.format(dataset_name))
        return getattr(module, "MVSDataset")
    except ImportError:
        pass
    # 回退到各方法自带的 datasets/
    module_name = 'datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    return getattr(module, "MVSDataset")
