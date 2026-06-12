import torch.nn as nn
import torch
import torchvision
import copy
import torch.nn.functional as F

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


'''
2 Models Available:
   - SER_AlexNet     : AlexNet model from pyTorch (CNN features layer + FC classifier layer)
   - SER_AlexNet_GAP : Fully-Convolutional model with AlexNet features layer + global average pooling (GAP) classifier
                        layer
'''



class SER_AlexNet(nn.Module):
    """
    Reference:
    https://pytorch.org/docs/stable/torchvision/models.html#id1

    AlexNet model from torchvision package. The model architecture is slightly
    different from the original model.
    See: AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.


    Parameters
    ----------
    num_classes : int
    in_ch   : int
        The number of input channel.
        Default AlexNet input channels is 3. Set this parameters for different
            numbers of input channels.
    pretrained  : bool
        To initialize the weight of AlexNet.
        Set to 'True' for AlexNet pre-trained weights.

    Input
    -----
    Input dimension (N,C,H,W)

    N   : batch size
    C   : channels
    H   : Height
    W   : Width

    Output
    ------
    logits (before Softmax)

    """


    def __init__(self,num_classes=4, in_ch=3, pretrained=True):
        super(SER_AlexNet, self).__init__()

        model = torchvision.models.alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool  = model.avgpool
        self.classifier = model.classifier

        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            init_layer(self.features[0])

        self.classifier[6] = nn.Linear(4096, num_classes)

        self._init_weights(pretrained=pretrained)
        
        print('\n<< SER AlexNet Finetuning model initialized >>\n')

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)

        return out

    def _init_weights(self, pretrained=True):

        init_layer(self.classifier[6])

        if pretrained == False:
            init_layer(self.features[0])
            init_layer(self.features[3])
            init_layer(self.features[6])
            init_layer(self.features[8])
            init_layer(self.features[10])
            init_layer(self.classifier[1])
            init_layer(self.classifier[4])




class SER_AlexNet_GAP(nn.Module):
    """
    Reference:
    https://pytorch.org/docs/stable/torchvision/models.html#id1

    AlexNet model from torchvision package. The model architecture is slightly
    different from the original model.
    See: AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.


    Parameters
    ----------
    num_classes : int
    in_ch   : int
        The number of input channel.
        Default AlexNet input channels is 3. Set this parameters for different
            numbers of input channels.
    pretrained  : bool
        To initialize the weight of AlexNet.
        Set to 'True' for AlexNet pre-trained weights.

    Input
    -----
    Input dimension (N,C,H,W)

    N   : batch size
    C   : channels
    H   : Height
    W   : Width

    Output
    ------
    logits (before Softmax)

    """


    def __init__(self,num_classes=4, in_ch=3, pretrained=True):
        super(SER_AlexNet_GAP, self).__init__()

        model = torchvision.models.alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool  = model.avgpool
        
        #Global average pooling layer
        self.classifier = nn.Sequential(
                                        nn.Conv2d(256, num_classes, kernel_size=(1,1)),
                                        nn.AvgPool2d(6)
                                        )

        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            init_layer(self.features[0])


        self._init_weights(pretrained=pretrained)
        
        print('\n<< SER AlexNet GAP model initialized >>\n')

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        out = self.classifier(x).squeeze(-1).squeeze(-1)

        return out

  
    def _init_weights(self, pretrained=True):

        init_layer(self.classifier[0])

        if pretrained == False:
            init_layer(self.features[0])
            init_layer(self.features[3])
            init_layer(self.features[6])
            init_layer(self.features[8])
            init_layer(self.features[10])


class SER_ResNet(nn.Module):
    def __init__(self, num_classes=4, in_ch=3, pretrained=True):
        super(SER_ResNet, self).__init__()
        if pretrained:
            try:
                self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            except AttributeError:
                self.model = torchvision.models.resnet18(pretrained=True)
        else:
            self.model = torchvision.models.resnet18(pretrained=False)
        
        # Modify first conv layer if in_ch != 3
        if in_ch != 3:
            self.model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            init_layer(self.model.conv1)
            
        # Modify classification layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        init_layer(self.model.fc)
        print('\n<< SER ResNet18 model initialized >>\n')

    def forward(self, x):
        return self.model(x)


class SER_DenseNet(nn.Module):
    def __init__(self, num_classes=4, in_ch=3, pretrained=True):
        super(SER_DenseNet, self).__init__()
        if pretrained:
            try:
                self.model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
            except AttributeError:
                self.model = torchvision.models.densenet121(pretrained=True)
        else:
            self.model = torchvision.models.densenet121(pretrained=False)
        
        # Modify first conv layer if in_ch != 3
        if in_ch != 3:
            self.model.features.conv0 = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            init_layer(self.model.features.conv0)
            
        # Modify classification layer
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        init_layer(self.model.classifier)
        print('\n<< SER DenseNet121 model initialized >>\n')

    def forward(self, x):
        return self.model(x)


class SER_EfficientNet(nn.Module):
    def __init__(self, num_classes=4, in_ch=3, pretrained=True):
        super(SER_EfficientNet, self).__init__()
        if pretrained:
            try:
                self.model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
            except AttributeError:
                self.model = torchvision.models.efficientnet_b0(pretrained=True)
        else:
            self.model = torchvision.models.efficientnet_b0(pretrained=False)
        
        # Modify first conv layer if in_ch != 3
        if in_ch != 3:
            self.model.features[0][0] = nn.Conv2d(in_ch, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            init_layer(self.model.features[0][0])
            
        # Modify classification layer
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        init_layer(self.model.classifier[1])
        print('\n<< SER EfficientNet-B0 model initialized >>\n')

    def forward(self, x):
        return self.model(x)


class SER_MobileNet(nn.Module):
    def __init__(self, num_classes=4, in_ch=3, pretrained=True):
        super(SER_MobileNet, self).__init__()
        if pretrained:
            try:
                self.model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
            except AttributeError:
                self.model = torchvision.models.mobilenet_v2(pretrained=True)
        else:
            self.model = torchvision.models.mobilenet_v2(pretrained=False)
        
        # Modify first conv layer if in_ch != 3
        if in_ch != 3:
            self.model.features[0][0] = nn.Conv2d(in_ch, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            init_layer(self.model.features[0][0])
            
        # Modify classification layer
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        init_layer(self.model.classifier[1])
        print('\n<< SER MobileNet-V2 model initialized >>\n')

    def forward(self, x):
        return self.model(x)
