from utils.transforms import get_transforms
import os
from avalanche.benchmarks.utils import (
    make_classification_dataset,
    classification_subset,
)
from shift_dev.types import Keys, WeathersCoarse, TimesOfDayCoarse
from shift_dev.utils.backend import HDF5Backend, ZipBackend
from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark

from shift_dev.dataloader.shift_dataset import SHIFTDataset


_WEATHERS_SEQUENCE = [WeathersCoarse.cloudy,
                      WeathersCoarse.overcast,
                      WeathersCoarse.rainy,
                      WeathersCoarse.foggy]

_TIMEOFDAY_SEQUENCE = [TimesOfDayCoarse.dawn_dusk,
                       TimesOfDayCoarse.night]


class _SHIFTClassificationTargets:
    def __init__(self, shift: SHIFTDataset):
        self.shift = shift

    def __len__(self):
        return len(self.shift)

    def __getitem__(self, idx):
        return self.shift.scalabel_datasets[f"{self.shift.views_to_load[0]}/det_2d"].get_classification_target(idx)


class _SHIFTClassificationDataset(SHIFTDataset):
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
    
def _get_weather_sets(cfg):
    train_sets = []
    val_sets = []
    
    transforms_test = get_transforms(cfg, train=False)

    for weather in _WEATHERS_SEQUENCE:
        print(
            f"Loading {weather} weather {TimesOfDayCoarse.daytime} time of day data...")

        train_sets.append(_SHIFTClassificationDataset(split='train',
                                        data_root=cfg['data_root'],
                                        transforms=transforms_test,
                                        weathers_coarse=[weather],
                                        timeofdays_coarse=[
                                            TimesOfDayCoarse.daytime],
                                        backend=ZipBackend(),
                                        classification_img_size=cfg['img_size']))
        print(
            f"{weather} weather {TimesOfDayCoarse.daytime} time data train split size: {len(train_sets[-1])}")

        val_sets.append(_SHIFTClassificationDataset(split='val',
                                            data_root=cfg['data_root'],
                                            transforms=transforms_test,
                                            weathers_coarse=[weather],
                                            timeofdays_coarse=[
                                                TimesOfDayCoarse.daytime],
                                            backend=ZipBackend(),
                                            classification_img_size=cfg['img_size']))
        print(
            f"{weather} weather {TimesOfDayCoarse.daytime} time data val split size: {len(val_sets[-1])}")
    return train_sets, val_sets

def _get_timeofday_sets(cfg):
    train_sets = []
    val_sets = []
    
    transforms_test = get_transforms(cfg, train=False)

    for timeofday in _TIMEOFDAY_SEQUENCE:
        print(f"Loading {WeathersCoarse.clear} weather {timeofday} time of day data...")

        train_sets.append(_SHIFTClassificationDataset(split='train',
                                        data_root=cfg['data_root'],
                                        transforms=transforms_test,
                                        weathers_coarse=[WeathersCoarse.clear],
                                        timeofdays_coarse=[timeofday],
                                        backend=ZipBackend(),
                                        classification_img_size=cfg['img_size']))
        print(f"{WeathersCoarse.clear} weather {timeofday} time data train split size: {len(train_sets[-1])}")

        val_sets.append(_SHIFTClassificationDataset(split='val',
                                            data_root=cfg['data_root'],
                                            transforms=transforms_test,
                                            weathers_coarse=[WeathersCoarse.clear],
                                            timeofdays_coarse=[timeofday],
                                            backend=ZipBackend(),
                                            classification_img_size=cfg['img_size']))
        print(f"{WeathersCoarse.clear} weather {timeofday} time data val split size: {len(val_sets[-1])}")
    return train_sets, val_sets

def _get_domains_mix_sets(cfg):
    train_sets = []
    val_sets = []
    
    transforms_test = get_transforms(cfg, train=False)
    
    # source domain, but validation split
    val_sets.append(_SHIFTClassificationDataset(split='val',
                                                data_root=cfg['data_root'],
                                                transforms=transforms_test,
                                                weathers_coarse=[WeathersCoarse.clear],
                                                timeofdays_coarse=[TimesOfDayCoarse.daytime],
                                                backend=ZipBackend(),
                                                classification_img_size=cfg['img_size']))
    print(f"{WeathersCoarse.clear} weather {TimesOfDayCoarse.daytime} time data val split size: {len(val_sets[-1])}")
    
    times_seq = [TimesOfDayCoarse.daytime, *_TIMEOFDAY_SEQUENCE]
    weathers_seq = [WeathersCoarse.clear, *_WEATHERS_SEQUENCE]

    for timeofday in times_seq:
        for weather in weathers_seq:
            print(f"Loading {weather} weather {timeofday} time of day data...")

            train_sets.append(_SHIFTClassificationDataset(split='train',
                                            data_root=cfg['data_root'],
                                            transforms=transforms_test,
                                            weathers_coarse=[weather],
                                            timeofdays_coarse=[timeofday],
                                            backend=ZipBackend(),
                                            classification_img_size=cfg['img_size']))
            print(f"{weather} weather {timeofday} time data train split size: {len(train_sets[-1])}")
            
    return train_sets, val_sets

# TODO: add clear and daytime as source experience for eval

def get_shift_benchmark(cfg):
    train_sets = []
    val_sets = []
    if cfg['benchmark'] == "shift_weather":
        train_sets, val_sets = _get_weather_sets(cfg)
    elif cfg['benchmark'] == "shift_timeofday":
        train_sets, val_sets = _get_timeofday_sets(cfg)
    elif cfg['benchmark'] == 'shift_mix':
        train_sets, val_sets = _get_domains_mix_sets(cfg)
    else:
        raise ValueError("Unknown type of shift benchmark")

    # transform_groups = dict(
    #     train=(None, None),
    #     eval=(None, None),
    # )

    train_exps_datasets = []
    for i, train_set in enumerate(train_sets):
        train_dataset_avl = make_classification_dataset(
            train_set,
            # transform_groups=transform_groups,
            initial_transform_group="train",
            task_labels=i,
            
        )

        train_exps_datasets.append(
            classification_subset(train_dataset_avl)
        )

    val_exps_datasets = []
    for i, val_set in enumerate(val_sets):
        val_dataset_avl = make_classification_dataset(
            val_set,
            # transform_groups=transform_groups,
            initial_transform_group="eval",
            task_labels=i
        )

        val_exps_datasets.append(
            classification_subset(val_dataset_avl)
        )

    return create_multi_dataset_generic_benchmark(train_datasets=train_exps_datasets,
                                                  test_datasets=val_exps_datasets,
                                                  # train_transform=None,
                                                  # eval_transform=None
                                                  )
