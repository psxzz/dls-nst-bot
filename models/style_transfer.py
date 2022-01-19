import copy
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as tt
import torchvision.models as models
from torchvision.utils import save_image


def load_image(image_path, device):
    image = Image.open(image_path).convert('RGB')
    imsize = 512 if device == torch.device('cuda') else 256

    image_transforms = tt.Compose(
        [
            tt.Resize(imsize),
            tt.CenterCrop(imsize),
            tt.ToTensor(),
        ]
    )
    image = image_transforms(image)[:3, :, :].unsqueeze(0)

    return image.to(device)


def get_gram_matrix(tensor):
    c, n, h, w = tensor.size()
    tensor = tensor.view(c * n, h * w)
    gram = torch.mm(tensor, tensor.t())

    return gram.div(c * n * h * w)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = None

    def forward(self, input):
        # MSE was replaced by l1_loss due to lesser penalties for outliers
        self.loss = F.l1_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = get_gram_matrix(target_feature).detach()
        self.loss = None

    def forward(self, input):
        G = get_gram_matrix(input)
        # MSE was replaced by l1_loss due to lesser penalties for outliers
        self.loss = F.l1_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, image):
        return (image - self.mean) / self.std


class NST:
    def __init__(self, device=None):
        self.device = torch.device('cpu') if device is None else device

        self.cnn = models.vgg19_bn(pretrained=True).features.to(self.device).eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        # Style and content layers were moved
        # from convolutional to activation layers
        self.style_layers_default = ['relu_1', 'relu_2', 'relu_3', 'relu_4', 'relu_5']
        self.content_layers_default = ['relu_4']

    def model_init(self, style_image, content_image, style_layers=None, content_layers=None):
        # Initialize layers if its none
        if style_layers is None:
            style_layers = self.style_layers_default

        if content_layers is None:
            content_layers = self.content_layers_default

        # Copy model
        cnn_copy = copy.deepcopy(self.cnn)

        normalization = Normalization(self.mean, self.std).to(self.device)

        content_losses = []
        style_losses = []

        # Build new model
        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn_copy.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
                # Replacing MaxPool to AvgPool for higher quality results
                layer = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError(
                    'Unrecognized layer {}'.format(layer.__class__.__name__)
                )

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module('content_loss_{}'.format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module('style_loss_{}'.format(i), style_loss)
                style_losses.append(style_loss)

        # Remove useless layers
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[: (i + 1)]

        return model, style_losses, content_losses

    def transform_image(self, content_path, style_path, save_path, n_epochs=500, 
                            style_weight=1000000, content_weight=1, 
                            style_layers=None, content_layers=None):
        if self.device == torch.device('cuda'):
            torch.cuda.empty_cache()

        # Loading images
        style_image = load_image(style_path, self.device)
        content_image = load_image(content_path, self.device)

        # Building model
        model, style_losses, content_losses = self.model_init(
            style_image, content_image, style_layers, content_layers
        )
        model.requires_grad_(False).to(self.device)

        # Building target
        target = content_image.clone().requires_grad_(True).to(self.device)

        # optimizer initialization
        # LBFGS was replaced by Adam due to stability
        # when training with a large style_weight and a large n_epoch
        optimizer = optim.Adam([target.requires_grad_()], lr=0.03)

        # Main train loop
        for epoch in range(1, n_epochs+1):
            with torch.no_grad():
                target.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(target)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            optimizer.step()

        with torch.no_grad():
            target.data.clamp_(0, 1)

        save_image(target, save_path)


def create_model():
    global nst
    nst = NST()


async def transform(content_path, style_path, save_path):
    nst.transform_image(content_path, style_path, save_path)
