import os
__all__ = ["BP_DIR", "CLEAN_TRAIN_DATA", "EXPERIMENTAL_DATA"]
FILE_DIR = os.path.dirname(__file__)
BP_DIR = os.path.join(FILE_DIR, 'bp_store')
CLEAN_DATA_DIR = os.path.join(FILE_DIR, 'CleanData')
CLEAN_DATA_GROUPED = os.path.join(CLEAN_DATA_DIR, 'grouped')
CLEAN_DATA_UNGROUPED = os.path.join(CLEAN_DATA_DIR, 'ungrouped')
CLEAN_TRAIN_DATA = os.path.join(CLEAN_DATA_GROUPED, 'ready')
EXPERIMENTAL_DATA = os.path.join(FILE_DIR, 'experimental_data')