from os import system
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as tt
import torchvision.models as models


def load_image(image_path, max_size=512, shape=None):
    image = Image.open(image_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    image_transforms = tt.Compose([
            # tt.Resize((size, int(1.5 * size))),
            tt.Resize(size),
            tt.ToTensor(),
            tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image = image_transforms(image)[:3, :, :].unsqueeze(0)

    return image


def im_convert(tensor):
    image = tensor.to('cpu').clone().detach()
    image = np.rollaxis(image.numpy().squeeze(), 0, 3)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

    image = image.clip(0, 1)

    return image

def inverse_normalize(tensor):

    inv_trans = tt.Compose([
        tt.Normalize((0., 0., 0.),(1/0.229, 1/0.224, 1/0.225)),
        tt.Normalize((-0.485, -0.456, -0.406),(1., 1., 1.))
    ])

    return inv_trans(tensor)


class NST:
    def __init__(self, device=torch.device('cpu'), model_init=None, weights=None, style_weight=1e3, content_weight=1e4, layers=None):
        self.device = device
        self.imsize = 512 if self.device == torch.device('cuda') else 256

        if weights is None:
            self.style_weights = {
                'conv1_1': 0.75,
                'conv2_1': 0.5,
                'conv3_1': 0.25,
                'conv4_1': 0.2,
                'conv5_1': 0.2,
            }

        self.style_weight = style_weight
        self.content_weight = content_weight

        self.model = self.model_init() if model_init is None else model_init()

        self.layers = layers

    def model_init(self):
        torch.utils.model_zoo.load_url(
            'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', model_dir='models/'
        )
        cnn = models.vgg19()
        cnn.load_state_dict(torch.load('models/vgg19-dcbb9e9d.pth'))
        for p in cnn.parameters():
            p.requires_grad_(False)

        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                cnn.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        return cnn

    def get_features(self, image):
        if self.layers is None:
            self.layers = {
                '0': 'conv1_1',  # style layers
                '5': 'conv2_1',
                '10': 'conv3_1',
                '19': 'conv4_1',
                '28': 'conv5_1',
                '21': 'conv4_2',  # content layer
            }

        features = {}
        x = image
        for name, layer in enumerate(self.model.features):
            x = layer(x)
            if str(name) in self.layers:
                features[self.layers[str(name)]] = x

        return features

    def get_gram_matrix(self, tensor):
        c, n, h, w = tensor.size()
        tensor = tensor.view(n, h * w)
        gram = torch.mm(tensor, tensor.t())

        return gram

    def transform_image(self, content_path, style_path, n_epochs, from_content=False):
        if self.device == torch.device('cuda'):
            torch.cuda.empty_cache()

        content = load_image(content_path, self.imsize).to(self.device)
        style = load_image(style_path, self.imsize).to(self.device)

        self.model.to(self.device).eval()

        style_features = self.get_features(style)
        content_features = self.get_features(content)

        style_gram_matrices = {
            layer: self.get_gram_matrix(style_features[layer]) for layer in style_features
        }

        if not from_content:
            target = torch.randn_like(content).requires_grad_(True).to(self.device)
        else:
            target = content.clone().requires_grad_(True).to(self.device)

        optimizer = optim.LBFGS([target])

        epoch = [0]
        while epoch[0] <= n_epochs:

            def closure():
                optimizer.zero_grad()

                # Getting target features
                target_features = self.get_features(target)

                # Computing content loss
                content_loss = torch.mean(
                    (target_features['conv4_2'] - content_features['conv4_2']) ** 2
                )

                # Computing style loss
                style_loss = 0
                for layer in self.style_weights:
                    target_feature = target_features[layer]
                    target_gram_matrix = self.get_gram_matrix(target_feature)

                    _, c, h, w = target_feature.shape
                    style_gram_matrix = style_gram_matrices[layer]

                    style_loss_per_layer = self.style_weights[layer] * torch.mean(
                        (target_gram_matrix - style_gram_matrix) ** 2
                    )

                    style_loss += style_loss_per_layer / (c * h * w)

                content_loss = self.content_weight * content_loss
                style_loss = self.style_weight * style_loss

                loss_fn = content_loss + style_loss
                loss_fn.backward(retain_graph=True)

                epoch[0] += 1
                if epoch[0] % 50 == 0:
                    system('clear')
                    print(f"Epoch: {epoch[0]} / {n_epochs}")   
                    print(f"Total loss: {round(loss_fn.item(), 4)} (Style: {round(style_loss.item(), 2)}, Content: {round(content_loss.item(), 2)})")

                return content_loss + style_loss

            optimizer.step(closure)
                
        print('Transform complete, checkout \'results\' folder')

        return target
