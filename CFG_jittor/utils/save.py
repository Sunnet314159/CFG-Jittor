import os
import jittor as jt
import numpy as np
from PIL import Image

def save_model(model, path='./saved_models/Classifier_free_DDIM_MNIST.h5'):
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')
    jt.save(model.state_dict(), path)

def load_model(path='./saved_models/Classifier_free_DDIM_MNIST.h5',model=None):
    state_dict = jt.load(path)
    model.load_state_dict(state_dict)
    return model
