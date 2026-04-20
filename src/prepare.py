import os
import sys
import time
import math
import argparse
import pickle
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET_MINUTES = 30            # training time budget in minutes
MAX_LOOPS = 30            # maximum number of loops to run (safety check)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = "/mnt/ocean_storage/users/zzhao"
# BASE_DIR = "/mnt/ocean_storage/data/Chest_Xray/preprocessed_512/"
DATA_DIR = os.path.join(BASE_DIR, "GRAZPEDWRI-DX")
# DATA_DIR = os.path.join(BASE_DIR, "ISIC2019")
# If you need to save external publicaly available data, processed intermediate feature, processed images, additional pre-trained weight,
# or any kinds of new files, please use this EXT_DATA_DIR instead of current project directory.
EXT_DATA_DIR = "/mnt/ocean_storage/users/zzhao" 
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
