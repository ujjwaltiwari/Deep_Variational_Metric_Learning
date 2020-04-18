import torch
import torch.nn as nn
from torchvision.models.googlenet import GoogLeNet
from torchvision.models.utils import load_state_dict_from_url

class GoogLeNetAvgPool(GoogLeNet):
    def _forward(self,x):
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)

        return x

    def forward(self,x):
        x = self._transform_input(x)
        x = self._forward(x)
        return x

def googlenet(device = None):
    if not device:
        device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = load_state_dict_from_url('https://download.pytorch.org/models/googlenet-1378be20.pth',map_location = device)
    kwargs = {
                "transform_input" : True,
                "aux_logits" : True,
                "init_weights" : False,
            }
    model = GoogLeNetAvgPool(**kwargs)
    model.load_state_dict(checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
