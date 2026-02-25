from fastapi import FastAPI, UploadFile, File, HTTPException

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models