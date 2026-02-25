import copy
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from config.defaults import get_cfg
from dataset.ee4d_motion_dataset import EE4D_Motion_DataModule
from module.uem_module import UEM_Module
from module.ema import EMA