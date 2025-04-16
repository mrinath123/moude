from mmseg.registry import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class CityCDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(CityCDataset, self).__init__(
            img_suffix='_leftImg8bit.png',
            seg_map_suffix='_gtFine_labelTrainIds.png',
            **kwargs)
        self.valid_mask_size = [1080, 1920]