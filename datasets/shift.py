import os

from .shift_dev import SHIFTDataset
from .shift_dev.types import Keys


class _SHIFTClassificationTargets:
    def __init__(self, shift: SHIFTDataset):
        self.shift = shift

    def __len__(self):
        return len(self.shift)

    def __getitem__(self, idx):
        return self.shift.scalabel_datasets[f"{self.shift.views_to_load[0]}/det_2d"].get_classification_target(idx)


class SHIFTClassificationDataset(SHIFTDataset):
    NUM_CLASSES = 6

    def __init__(self, data_root, transforms = None, **kwargs) -> None:
        data_root = os.path.join(data_root, "shift")
        self.transforms = transforms
        
        super().__init__(
            views_to_load=["front"],
            keys_to_load=[
                Keys.images,
                Keys.boxes2d,
                Keys.boxes2d_classes
            ],
            classification=True,
            data_root=data_root,
            **kwargs
        )

    def __getitem__(self, idx: int):
        # target = self.targets[idx]
        sample = super().__getitem__(idx)
        target = sample[self.views_to_load[0]][Keys.boxes2d_classes].item()
        img = sample[self.views_to_load[0]][Keys.images][0]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    @property
    def targets(self):
        """
        Get a list of all category ids, required for Avalanche.
        """
        return _SHIFTClassificationTargets(self)

    @property
    def classes(self):
        config = self.scalabel_datasets[f"{self.views_to_load[0]}/det_2d"].cfg
        return [category.name for category in config.categories]