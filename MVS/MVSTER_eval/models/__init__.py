import importlib


def find_model_def(model_name):
    """查找模型定义"""
    module_name = 'models.{}'.format(model_name)
    module = importlib.import_module(module_name)
    return getattr(module, "MVS4net")
