from .classification.cladc import *
from .classification.cladc_utils import get_matching_classification_set, get_cladc_domain_sets
from .detection.cladd import *
from .utils.test_cladc import test_cladc, AMCAtester
from .utils.meta import SODA_DOMAINS, SODA_CATEGORIES

try:
    from .detection.cladd_detectron import register_cladd_detectron
except ModuleNotFoundError:
    print("[INFO] No Detectron installation found, continuing without.")
