import torch
from torch import nn
import torchvision.models as models

class RetinalModel(nn.Module):
    def __init__(self, 
        model: str="resnet50", 
        pretrained: bool=False, 
        num_classes: int=7,
    ):
        super().__init__()
        resnest_lst = torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)

        if model in resnest_lst:
            self.encoder = torch.hub.load('zhanghang1989/ResNeSt', model, pretrained=pretrained)
        else:
            self.encoder = getattr(models, model)(pretrained=pretrained)

        in_features = self.encoder.fc.in_features
        # disable original fc layer
        self.encoder.fc = nn.Identity()
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    def forward(self, X):
        X = self.encoder(X)
        return self.fc(X)

    def extract_features(self, X):
        return self.encoder(X)