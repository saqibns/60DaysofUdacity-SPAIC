import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm


DEVICE = 'cuda'
VGG_LAYERS = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
              '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
CONTENT_IMAGE = '/home/saqib/Projects/ArtWork/style_transfer/images/content/mcu-1-iron-man.jpg'
STYLE_IMAGE = '/home/saqib/Projects/ArtWork/style_transfer/images/style/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'
LAYER_WEIGHTS = {'conv1_1': 1.0, 'conv2_1': 0.75, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
CONTENT_LAYERS = ['conv4_2']
IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS = [0.229, 0.224, 0.225]
IMAGENET_MEANS_INV = [-0.485, -0.456, -0.406]
IMAGENET_STDS_INV = [1./0.229, 1./0.224, 1./0.225]
ALPHA = 1.0
BETA = 1e5
STEPS = 5000
SAVE_EVERY = 50
LR = 0.003
IMG_SAVE_PATH = '/home/saqib/Projects/ArtWork/style_transfer/images/results'
EXPERIMENT_NUMBER = 1


class Hook:
    def __init__(self, module):
        self.out = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, mod, inp, out):
        self.out = out

    def close(self):
        self.hook.close()


def load_image_tensor(image_path, size=(400, 400)):
    img = Image.open(image_path)
    tfms = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
            ])
    img_tensor = tfms(img)
    return img_tensor.unsqueeze(0)


def save_tensor_as_img(tensor, dirpath, image_name):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # denorm = transforms.Normalize(IMAGENET_MEANS_INV, IMAGENET_STDS_INV)
    tensor = tensor.detach().cpu()
    # tensor = denorm(tensor.squeeze(0))
    # Put channels at the end, convert to numpy and rescale
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    tensor = (tensor.numpy() * np.array(IMAGENET_STDS) + np.array(IMAGENET_MEANS)) * 255.0
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    image.save(f'{dirpath}/{image_name}.jpg', 'JPEG')


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


def mse(tensor1, tensor2):
    # Calculates the sum of squared differences
    sq_diff = (tensor1 - tensor2) ** 2
    return sq_diff.mean()


def compute_content_loss(image_features, content_features):
    return ssd(image_features, content_features)


def compute_normalized_gram_matrix(features):
    _, c, w, h = features.shape
    features = features.view(c, w * h)
    # Multiply the feature matrix with its transpose
    gram = torch.mm(features, features.t()) / (c * w * h)
    return gram


def compute_style_loss(image_gram_matrices, style_gram_matrices, weights, device):
    num_matrices = len(image_gram_matrices)
    losses = torch.zeros(num_matrices).to(device)
    for i in range(len(image_gram_matrices)):
        losses[i] = weights[i] * ssd(image_gram_matrices[i], style_gram_matrices[i])

    return losses.sum()


if __name__ == '__main__':
    content_image = load_image_tensor(CONTENT_IMAGE)
    style_image = load_image_tensor(STYLE_IMAGE)
    vgg = load_model(models.vgg19)
    vgg = vgg.features
    vgg.to(DEVICE)
    hooks = dict()
    # Register hooks
    for key, mod in vgg._modules.items():
        if key in VGG_LAYERS:
            hooks[VGG_LAYERS[key]] = Hook(mod)

    content_image = content_image.to(DEVICE)
    style_image = style_image.to(DEVICE)
    result = content_image.clone()
    result.requires_grad = True

    optimizer = torch.optim.Adam([result], lr=LR)

    style_weights = [LAYER_WEIGHTS[lyr] for lyr in STYLE_LAYERS]
    style_weights = torch.FloatTensor(style_weights).to(DEVICE)

    # Extract content features
    _ = vgg(content_image)
    content_feats = hooks[CONTENT_LAYERS[0]].out

    # Extract style features
    _ = vgg(style_image)
    style_gram_matrices = [compute_normalized_gram_matrix(hooks[lyr].out) for lyr in STYLE_LAYERS]

    # Optimize
    pbar = tqdm(range(STEPS), desc='0.0')
    for i in pbar:
        optimizer.zero_grad()
        _ = vgg(result)
        result_content_feats = hooks[CONTENT_LAYERS[0]].out
        result_gram_matrices = [compute_normalized_gram_matrix(hooks[lyr].out) for lyr in STYLE_LAYERS]
        content_loss = compute_content_loss(result_content_feats, content_feats)
        style_loss = compute_style_loss(result_gram_matrices, style_gram_matrices, style_weights, DEVICE)
        total_loss = ALPHA * content_loss + BETA * style_loss
        # print('content:', content_loss.item(), 'style:', style_loss.item())
        total_loss.backward()
        optimizer.step()
        pbar.set_description(str(round(total_loss.item(), 3)))

        if (i + 1) % SAVE_EVERY == 0:
            save_tensor_as_img(result.clone(), f'{IMG_SAVE_PATH}/{EXPERIMENT_NUMBER:03d}', f'{i+1:04d}')
