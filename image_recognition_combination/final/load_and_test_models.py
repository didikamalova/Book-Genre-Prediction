import torch
import numpy as np
from utils import load_dataset, load_labels, evaluate
from combined_models import OneParam, ThirtyParam, FullyConnected


def load_model(model_path, Model):
    model = Model()
    model.load_state_dict(torch.load(model_path))
    return model


test_image_dir = 'final_outputs/test_image_output.csv'
test_title_dir = 'final_outputs/test_title_output.csv'
test_y_dir = 'final_outputs/test_labels.csv'

test_x1 = torch.from_numpy(load_dataset(test_image_dir))
test_x2 = torch.from_numpy(load_dataset(test_title_dir))
test_y = torch.from_numpy(load_labels(test_y_dir)).type(torch.LongTensor)

comb_model = load_model('./saved_comb_models/comb2.pth', ThirtyParam)

outputs = comb_model(test_x1, test_x2)
evaluate(outputs, test_y, "test")
np.savetxt('./final_outputs/test_comb2_output.csv', outputs.detach().numpy(), delimiter=',')
