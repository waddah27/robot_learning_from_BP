import os

__all__ = [
    "CLEAN_TRAIN_DATA", 
    "EXPERIMENTAL_DATA",
    ]

_file_dir = os.path.dirname(__file__)

_clean_data_dir = os.path.join(_file_dir, 'CleanData')
_clean_data_grouped = os.path.join(_clean_data_dir, 'grouped')
_clean_data_ungrouped = os.path.join(_clean_data_dir, 'ungrouped')
CLEAN_TRAIN_DATA = os.path.join(_clean_data_grouped, 'ready')
EXPERIMENTAL_DATA = os.path.join(_file_dir, 'experimental_data')


