
'''
this script is the same as stylize.py except that the stylize.py was using the style-model weights and taking in input only.

what if there is a style that i want to use and there is no preset available for that?

this script is the solution. Live training. I have optimised it so that i can process content image and take in style image of choice and process it fast . Although the results maynot be as accurate cause it's training on-device cpu and in very less time.  

'''



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from utils import display_stylized_image
from gram_matrix import gram_matrix
from torchvision import transforms
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

def load_image(img_path, max_size=400, shape=(224, 224)):
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image



def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
        if name == 'avgpool':
            x = F.adaptive_avg_pool2d(x, (7, 7))  # Ensure the shape is correct for the linear layers
            break
    return features


def stylize_image(content_image, style_image, steps_to_run_for = 2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_image = load_image(content_image, shape=(224, 224)).to(device)
    style_image = load_image(style_image, shape=content_image.shape[-2:]).to(device)
    print("Content Image Shape: ", content_image.shape)
    print("Style Image Shape: ", style_image.shape)



    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
    for param in vgg.parameters():
        param.requires_grad = False
    vgg.to(device)

    content_features = get_features(content_image, vgg)
    style_features = get_features(style_image, vgg)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content_image.clone().requires_grad_(True).to(device)

    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}

    content_weight = 1
    style_weight = 1e9

    optimizer = optim.Adam([target], lr=0.003)
    steps = steps_to_run_for

    for ii in range(1, steps+1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss / (d * h * w)
        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    stylized_image = target.detach().cpu().numpy().squeeze()
    stylized_image = stylized_image.transpose(1,2,0)
    stylized_image = stylized_image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    stylized_image = stylized_image.clip(0, 1)

    display_stylized_image(stylized_image)

    return stylized_image

# Example usage:
content_image_path = 'samples/cat.jpg'
style_image_path ='presets/mosaic.jpg'
stylized_image_bytes = stylize_image(content_image_path, style_image_path)
# print(stylized_image_bytes.shape)
# print(stylized_image_bytes.dtype)
# print(stylized_image_bytes)
