# coding: UTF-8
import torch.nn as nn
from transformers import RobertaModel


class RCModel(nn.Module):
    def __init__(self, pretrained_path):
        super(RCModel, self).__init__()
        self.codebert = RobertaModel.from_pretrained(pretrained_path)
        self.f1 = nn.Sequential(
            nn.Linear(768, 2),
            nn.Softmax(1)
        )

    def forward(self, sentence):
        cls = self.codebert(sentence)[1]
        output = self.f1(cls)
        return output
