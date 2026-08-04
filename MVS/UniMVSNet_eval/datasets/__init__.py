import importlib


def find_dataset_def(dataset_name):
    """查找数据集定义，统一使用 general_eval"""
    module_name = 'datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    return getattr(module, "MVSDataset")
