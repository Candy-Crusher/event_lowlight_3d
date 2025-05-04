import importlib
from os import path as osp

from ..event_utils import scandir

# automatically scan and import model modules
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(model_folder)
    if v.endswith('_model.py')
]
# import all the model modules
_model_modules = [
    importlib.import_module(f'.{file_name}', package=__name__)
    for file_name in model_filenames
]

# 定义 opt 配置
opt = {
    "type": "SWINPad",
    "load_teacher": False,
    "pretrain_img_size": 192,
    "patch_size": 4,
    "in_chans": 5,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4,
    "qkv_bias": True,
    "qk_scale": None,
    "drop_rate": 0,
    "drop_path_rate": 0,
    "ape": False,
    "patch_norm": True,
    "use_checkpoint": False,
    "pretrained_checkpoint": "checkpoints/pr.pt",  # 设置 checkpoint 路径
    "pretrained_checkpoint_type": "event",
    "keep_patch_keys": False,
}

def create_model(opt=opt):
    """Create model.
    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_type = opt.pop('type')

    # dynamic instantiation
    for module in _model_modules:
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')

    model = model_cls(**opt)
    
    return model