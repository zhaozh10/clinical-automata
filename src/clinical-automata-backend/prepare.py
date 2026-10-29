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
MIN_LOOPS = 15             # minimum number of loops to run (safety check)
MAX_LOOPS = 30            # maximum number of loops to run (safety check)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Site-specific paths live in src/config.env (copied from src/config.env.example).
# Environment variables with the same name take precedence.
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.env")

def _load_config(path=CONFIG_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{os.path.abspath(path)} not found. Copy src/config.env.example to src/config.env and fill it in."
        )
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip("'\"")
    return {k: os.environ.get(k, v) for k, v in cfg.items()}

_CFG = _load_config()

BASE_DIR = _CFG["DATA_ROOT"]

DATA_DIR = {"wrist x-ray": os.path.join(BASE_DIR, "GRAZPEDWRI-DX"), # public
            "dermoscopy": os.path.join(BASE_DIR, "ISIC2019"), # public
            "chest x-ray (pneumothorax)": os.path.join(BASE_DIR, "siim-acr-pneumothorax"), #public
            "chest x-ray (14 diseases)": os.path.join(BASE_DIR, "NIH_ChestX-ray14"), # public
            "chest x-ray (anatomy segmentation)": os.path.join(BASE_DIR, "pax_ray++"), # public
            "pelvic ct (fracture)": os.path.join(BASE_DIR, "pengwin2026"), # public
            }


# DATA_DIR = os.path.join(BASE_DIR, "GRAZPEDWRI-DX")
# DATA_DIR = os.path.join(BASE_DIR, "ISIC2019")
# If you need to save external publicaly available data, processed intermediate feature, processed images, additional pre-trained weight,
# or any kinds of new files, please use this EXT_DATA_DIR instead of current project directory.
EXT_DATA_DIR = _CFG["EXT_DATA_ROOT"]
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
