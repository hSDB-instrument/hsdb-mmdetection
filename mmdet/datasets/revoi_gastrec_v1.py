from .coco import CocoDataset
from .registry import DATASETS

@DATASETS.register_module
class hSDBRevoiGastrecDatasetV1(CocoDataset):
    CLASSES = ('HarmonicAce_Head',
               'HarmonicAce_Body',
               'MarylandBipolarForceps_Head',
               'MarylandBipolarForceps_Wrist',
               'MarylandBipolarForceps_Body',
               'CadiereForceps_Head',
               'CadiereForceps_Wrist',
               'CadiereForceps_Body',
               'Medium-LargeClipApplier_Head',
               'Medium-LargeClipApplier_Wrist',
               'Medium-LargeClipApplier_Body',
               'Needle',
               'Needle-holder_Head',
               'Needle-holder_Body')