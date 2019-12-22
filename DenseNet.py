
import torch.nn as nn
import torchvision

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features
        #self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Softmax())
    def forward(self, x):
        x = self.densenet121(x)
        return x