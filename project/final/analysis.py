import numpy as np
import torch
from utils import load_labels, load_dataset, evaluate, analyze_class_accuracy
from combined_models import OneParam, ThirtyParam


def load_model(model_path, Model):
    model = Model()
    model.load_state_dict(torch.load(model_path))
    return model


