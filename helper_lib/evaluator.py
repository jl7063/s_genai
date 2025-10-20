import torch
from .data_loader import get_transforms

def prepare_eval_transform():
    return get_transforms(train=False)

def evaluate_model(model, device="cpu"):
    model.eval()
    return
