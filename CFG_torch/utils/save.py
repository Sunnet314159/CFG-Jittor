import os
import torch


def save_model(model, path='./saved_models/Classifier_free_DDIM_MNIST.h5'):
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')
    torch.save(model.state_dict(), path)

def load_model(path='./saved_models/Classifier_free_DDIM_MNIST.h5',model=None):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model
