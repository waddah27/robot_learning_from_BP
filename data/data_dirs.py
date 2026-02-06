import os
__all__ = ["BP_DIR", "CLEAN_TRAIN_DATA", "EXPERIMENTAL_DATA"]
FILE_DIR = os.path.dirname(__file__)
BP_DIR = os.path.join(FILE_DIR, 'bp_store')
CLEAN_DATA_DIR = os.path.join(FILE_DIR, 'CleanData')
CLEAN_DATA_PARALLEL = os.path.join(CLEAN_DATA_DIR, 'parallel')
CLEAN_TRAIN_DATA = os.path.join(CLEAN_DATA_PARALLEL, 'parallel_new')
EXPERIMENTAL_DATA = os.path.join(FILE_DIR, 'experimental_data')