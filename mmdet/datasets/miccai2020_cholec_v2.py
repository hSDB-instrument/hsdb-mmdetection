from .coco import CocoDataset
from .registry import DATASETS

@DATASETS.register_module
class hSDBCholeDataset(CocoDataset):
    CLASSES = ('AtraumaticGrasper_Head',
               'AtraumaticGrasper_Body',
               'Electrichook',
               'CurvedAtraumaticGrasper_Head',
               'CurvedAtraumaticGrasper_Body',
               'Suction-Irrigation',
               'ClipApplier(Metal)_Head',
               'ClipApplier(Metal)_Body',
               'Scissors_Head',
               'Scissors_Body',
               'Overholt_Head',
               'Overholt_Body',
               'Ligasure_Head',
               'Ligasure_Body',
               'ClipApplier(Hem-O-Lok)_Head',
               'ClipApplier(Hem-O-Lok)_Body',
               'Specimanbag')