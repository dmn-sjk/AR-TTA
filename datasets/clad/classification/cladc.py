from .cladc_utils import *
import os
from torch.utils.data import ConcatDataset

    
train_task_dicts = [{'date': '20191111', 'period': 'Daytime'},
                    {'date': '20191111', 'period': 'Night'},
                    {'date': '20191117', 'period': 'Daytime'},
                    {'date': '20191117', 'period': 'Night'},
                    {'date': '20191120', 'period': 'Daytime'},
                    {'date': '20191121', 'period': 'Night'}, ]

def get_cladc_single_train_task(root: str, task_id: int, transform: Callable = None, img_size: int = 224) -> CladClassification:
    task_dict = train_task_dicts[task_id]    
    match_fn = create_match_dict_fn(task_dict)

    # All training images are part of the original validation set of SODA10M.
    annot_file = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_val.json')
    train_set = get_matching_classification_set(root, annot_file, match_fn, img_size=img_size, transform=transform)
    train_set.chronological_sort()

    return train_set

def get_cladc_train(root: str, transform: Callable = None, img_size: int = 224, 
                    avalanche: bool = False) -> Sequence[CladClassification]:
    """
    Returns a sequence of training sets that are chronologically ordered, defined as in the ICCV '21 challenge.

    :param root: root path to the dataset
    :param transform: a callable transformation for the data images
    :param img_size: the width/height of the images, default is 224 by 224.
    :param avalanche: If true, this will return AvalancheDataset objects.
    """
    
    match_fn = (create_match_dict_fn(td) for td in train_task_dicts)

    # All training images are part of the original validation set of SODA10M.
    annot_file = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_val.json')
    train_sets = [get_matching_classification_set(root, annot_file, mf, img_size=img_size, transform=transform) for mf
                  in match_fn]

    for ts in train_sets:
        ts.chronological_sort()

    return train_sets


def get_cladc_val(root: str, transform: Callable = None, img_size: int = 64, avalanche=False) -> CladClassification:
    """
    Returns the default validation set of the ICCV '21 benchmark.
    """

    def val_match_fn_1(obj_id, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj_id]['image_id']]
        date = img_annot["date"]
        return not (date == "20191120" or date == "20191117" or date == "20191111" or
                    (date == "20191121" and img_annot['period'] == "Night"))

    def val_match_fn_2(obj_id, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj_id]['image_id']]
        time = img_annot['time']
        date = img_annot["date"]
        label = obj_dic[obj_id]['category_id']
        return label or (label and date == "20181015" and (time == '152030' or time == '160000'))

    annot_file_1 = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_val.json')
    annot_file_2 = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_train.json')

    val_set = ConcatDataset([
        get_matching_classification_set(root, annot_file_1, val_match_fn_1, img_size=img_size, transform=transform),
        get_matching_classification_set(root, annot_file_2, val_match_fn_2, img_size=img_size, transform=transform)])

    return val_set


def get_cladc_test(root: str, transform=None, img_size: int = 64, avalanche=False):
    """
    Returns the full test set of the CLAD-C benchmark. Some domains are overly represented, so for a fair
    evaluation see get_cladc_domain_test.
    """

    annot_file = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_test.json')
    test_set = get_matching_classification_set(root, annot_file, lambda *args: True, img_size=img_size,
                                               transform=transform)

    return test_set


def get_cladc_domain_test(root: str, transform: Callable = None, img_size: int = 64, avalanche=False):
    """
    Returns fine-grained domain sets of the test set for each available combination, note that not all the
    combinations of domains exist.
    """
    annot_file = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_test.json')
    test_sets = get_cladc_domain_sets(root, annot_file, ["period", "weather", "city", "location"],
                                      img_size=img_size, transform=transform)

    return [ts for ts in test_sets if len(ts) > 0]
