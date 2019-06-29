import torch
from torch import nn
from torchvision import models, transforms


DEVICE = 'cpu'
VGG_LAYERS = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
              '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}


class Hook:
    def __init__(self, module):
        self.out = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, mod, inp, out):
        self.out = out

    def close(self):
        self.hook.close()


def load_model(arch):
    # Load a pretrained torchvision model
    model = arch(pretrained=True)
    # We don't need gradients for the model
    for param in model.parameters():
        param.requires_grad = False

    return model


def ssd(tensor1, tensor2):
    # Calculates the sum of squared differences
    sq_diff = (tensor1 - tensor2) ** 2
    return sq_diff.sum()


def content_loss(image_features, content_features):
    return ssd(image_features, content_features)


def compute_normalized_gram_matrix(features):
    _, c, w, h = features.shape
    features = features.view(c, w * h)
    # Multiply the feature matrix with its transpose
    gram = torch.mm(features, features.t()) / (c * w * h)
    return gram


def style_loss(image_gram_matrices, style_gram_matrices, weights, device):
    num_matrices = len(image_gram_matrices)
    losses = torch.zeros(num_matrices).to(device)
    for i in range(len(image_gram_matrices)):
        losses[i] = weights[i] * ssd(image_gram_matrices[i], style_gram_matrices[i])

    return losses.sum()

