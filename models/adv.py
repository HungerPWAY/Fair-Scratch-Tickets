import torch
import torch.nn as nn
from torchvision import models
import pdb
from .resnet import BasicBlock, Bottleneck

from utils.builder import get_builder
from args import args


class ResNet(nn.Module):
    def __init__(self, builder, block, layers, num_classes=1, base_width=64):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")

        if args.first_layer_dense:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = builder.conv7x7(3, 64, stride=2, first_layer=True)

        self.bn1 = builder.batchnorm(64)
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, 64, layers[0])
        self.layer2 = self._make_layer(builder, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(builder, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)


    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(
                self.inplanes, planes * block.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x

class Encoder(nn.Module):
    def __init__ (self, hidden_size):
        super(Encoder, self).__init__()
        
        # Load pretrained resnet model
        #self.resnet = ResNet(get_builder(), BasicBlock, [2, 2, 2, 2], hidden_size)
        self.resnet = ResNet(get_builder(), Bottleneck, [3,4,6,3], hidden_size)
        # Remove the fully connected layers

        
        # Create our replacement layers
        # We reuse the in_feature size of the resnet fc layer for our first replacement layer = 2048 as of creation
        builder = get_builder()
        self.bn = builder.batchnorm(hidden_size, last_bn=True)

    def forward (self, images):
        # Get the expected output from the fully connected layers
        # Fn: AvgPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
        # Output: torch.Size([batch_size, 2048, 1, 1])
        features = self.resnet(images)

        # Resize the features for our linear function
        #features = features.view(features.size(0), -1)
        
        # Fn: Linear(in_features=2048, out_features=embed_size, bias=True)
        # Output: torch.Size([batch_size, embed_size])
        #features = self.linear(features)
        
        # Fn: BatchNorm1d(embed_size, eps=1e-05, momentum=0.01, affine=True)
        # Output: torch.Size([batch_size, embed_size])
        features = self.bn(features)
        
        return features

class Classifier(nn.Module):
    def __init__ (self, hidden_size, num_classes=1):
        super(Classifier, self).__init__()
        builder = get_builder()
        self.model = torch.nn.Sequential(
            builder.conv1x1(hidden_size, 512),
            nn.ReLU(),
            builder.conv1x1(512, 512),
            nn.ReLU(),
            builder.conv1x1(512, num_classes)
        )

    def forward (self, x):
        #x = x.view(-1, 512, 1, 1)
        out = self.model(x)
        out = out.view(out.shape[0],1)
        return out

class AdversarialHead(nn.Module):
    def __init__ (self, hidden_size):
        super(AdversarialHead, self).__init__()

        # self.model = torch.nn.Sequential(
        #     nn.Linear(in_features=hidden_size, out_features=512),
        #     nn.ReLU(),
        #     nn.Linear(in_features=512, out_features=512),
        #     nn.ReLU(),
        #     nn.Linear(in_features=512, out_features=1),
        # )
        builder = get_builder()
        self.model = torch.nn.Sequential(
            #builder.conv1x1(hidden_size, 512),
            #nn.LeakyReLU(negative_slope=0.1),
            builder.conv1x1(512, 256),
            nn.LeakyReLU(negative_slope=0.1),
            builder.conv1x1(256, 64),
            nn.LeakyReLU(negative_slope=0.1),
            #builder.conv1x1(128, 64),
            #nn.LeakyReLU(negative_slope=0.1),
            builder.conv1x1(64, 1)
        )

    def forward (self, x):

        #x = x.view(-1, 512, 1, 1)
        out = self.model(x)
        out_detached = self.model(x.detach())
        out = out.view(out.shape[0],1)
        out_detached = out_detached.view(out_detached.shape[0],1)
        return (out, out_detached)

class BaselineModel(nn.Module):
    def __init__ (self, hidden_size, num_classes=39):
        super(BaselineModel, self).__init__()

        self.encoder = Encoder(hidden_size)
        self.classifier = Classifier(hidden_size, num_classes)

    def forward (self, images):

        h = self.encoder(images)
        y = self.classifier(h)
        return y, None

    def sample (self, images):
        """
        Method to perform classification without computing
        adversarial head output.
          images: tensor of shape (batch_size, num_channels, height, width)
          return: tensor of shape (batch_size, num_classes)
        """
        h = self.encoder(images)
        y = self.classifier(h)
        return y

class OurModel(nn.Module):
    def __init__ (self, hidden_size = 512, num_classes=1):
        super(OurModel, self).__init__()

        self.encoder = Encoder(hidden_size)
        self.classifier = Classifier(hidden_size, num_classes)
        self.adv_head = AdversarialHead(hidden_size)


    # def forward (self, images, images_subset):
    def forward (self, images, protected_class_labels):

        # h_images = self.encoder(images)
        # y = self.classifer(h_images)
        # h_images_subset = self.encoder(images_subset)
        # a = self.adv_head(h_images_subset)

        h = self.encoder(images) # (batch_size, hidden_size)
        y = self.classifier(h) # (batch_size, num_classes)

        protected_class_encoded_images = h[protected_class_labels] 
        if protected_class_encoded_images.shape[0] == 0:
            return y, (None, None)
        
        a, a_detached = self.adv_head(protected_class_encoded_images)
        return y, (a, a_detached)

    def sample (self, images):
        """
        Method to perform classification without computing
        adversarial head output.
          images: tensor of shape (batch_size, num_channels, height, width)
          return: tensor of shape (batch_size, num_classes)
        """
        h = self.encoder(images)
        y = self.classifier(h)
        return y


