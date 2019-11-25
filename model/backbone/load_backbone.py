'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-11-25
@Email: xxxmy@foxmail.com
'''

from .vgg16 import vgg16
import torch.nn as nn


def load_backbonevgg16(pretrained=True,**kwargs):

    model=vgg16(pretrained=pretrained, progress=True, **kwargs)
    features=list(model.features)[:30]# the 30th layer of features is relu of conv5_3
    #freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad=False
    classifier=list(model.classifier)
    #del the last layer
    del classifier[6]
    #del dropout layer
    del classifier[5]
    del classifier[2]

    return nn.Sequential(*features),nn.Sequential(*classifier)