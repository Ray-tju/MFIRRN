import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import attention

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



class LLNet(nn.Module):

    def __init__(self, **kwargs):
        super(LLNet, self).__init__()

        self.encoder_fine = resnet34_module(pretrained=False)
        self.encoder_medium = resnet34_module(pretrained=False, x1_bool=1)
        self.encoder_coarse = resnet34_module(pretrained=False, x1_bool=1, x2_bool=1)

        self.concat_decoder = nn.Sequential(
            nn.Linear(186, 512),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, 186),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(186),
            nn.ELU(inplace=True),
            nn.Linear(186, 62),
        )

        self.attention = attention_1d.SELayer(186, 3)

        self.encoder_coarse.fc = nn.Sequential()

        self.encoder_fine.fc = nn.Sequential()
        self.encoder_fine.layer4 = nn.Sequential()
        self.encoder_fine.avgpool = nn.Sequential()

        self.encoder_medium.fc = nn.Sequential()
        self.encoder_medium.avgpool = nn.Sequential()

        self.max1 = nn.AvgPool2d(kernel_size=15, stride=15)
        self.max2 = nn.AvgPool2d(kernel_size=8, stride=8)

        self.conv_block1 = nn.Sequential(
            BasicConv(128, 2048, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(2048, 512, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(256, 2048, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(2048, 512, kernel_size=3, stride=1, padding=1, relu=True)
        )
        # self.conv_block3 = nn.Sequential(
        #     BasicConv(512, 2048, kernel_size=1, stride=1, padding=0, relu=True),
        #     BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True)
        # )

        self.fc_fine = nn.Sequential(nn.Linear(512, 256),
                                     nn.Dropout(p=0.3),
                                     nn.ELU(inplace=True),
                                     nn.Linear(256, 62))

        self.fc_medium = nn.Sequential(nn.Linear(512, 256),
                                       nn.Dropout(p=0.3),
                                       nn.ELU(inplace=True),
                                       nn.Linear(256, 62))

        self.fc_coarse = nn.Sequential(nn.Linear(512, 256),
                                       nn.Dropout(p=0.3),
                                       nn.ELU(inplace=True),
                                       nn.Linear(256, 62)
                                       )

    def forward(self, x, x1, x2):

        output_fine, output_fine_x1, _ = self.encoder_fine(x2)
        # print(111, output_fine_x1.size())
        output_fine = self.conv_block1(output_fine_x1)
        output_fine = self.max1(output_fine)
        output_fine = output_fine.view(output_fine.size(0), -1)
        output_fine = self.fc_fine(output_fine)

        output_medium, _, output_medium_x2 = self.encoder_medium(x1, output_fine_x1)
        output_medium = self.conv_block2(output_medium_x2)
        output_medium = self.max2(output_medium)
        output_medium = output_medium.view(output_medium.size(0), -1)
        output_medium = self.fc_medium(output_medium)

        output, _, _ = self.encoder_coarse(x, output_fine_x1, output_medium_x2)
        # output = self.conv_block3(output.view(output.size(0), -1, 1, 1))
        output = self.fc_coarse(output.view(output.size(0), -1))

        concat_input = torch.cat((output, output_medium, output_fine), dim=-1)
        # concat_out = self.attention(concat_input.unsqueeze(-1))
        concat_out = self.concat_decoder(concat_input.squeeze(-1))

        return output, output_medium, output_fine, concat_out




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, x1_bool=0, x2_bool=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.x1_bool = x1_bool
        self.x2_bool = x2_bool
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.x1_fusion_conv = nn.Sequential(
            BasicConv(256, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 128, kernel_size=3, stride=1, padding=1, relu=True),
            attention.SELayer(128, 16),
        )

        self.x2_fusion_conv = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 256, kernel_size=3, stride=1, padding=1, relu=True),
            attention.SELayer(256, 16),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, x1_input=0, x2_input=0):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = self.layer2(x)
        if self.x1_bool:
            x1 = torch.cat([x1, x1_input], dim=1)
            x1 = self.x1_fusion_conv(x1)
        x2 = self.layer3(x1)
        if self.x2_bool:
            x2 = torch.cat([x2, x2_input], dim=1)
            x2 = self.x2_fusion_conv(x2)
        x = self.layer4(x2)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, x1, x2


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model



def resnet34(**kwargs):
    model = LLNet(**kwargs)
    return model

def resnet34_module(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
