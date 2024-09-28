import torchvision.models as models
import torch.nn as nn
import timm
import torch


def get_model(model_name, num_classes):
    if model_name == 'densenet121':
        return get_densenet121(num_classes)
    elif model_name == 'vit':
        return get_vit(num_classes)


def get_densenet121(num_classes):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def get_vit(num_classes):
    model = timm.create_model('vit_base_patch16_224',
                              pretrained=True,
                              num_classes=num_classes)
    return model
