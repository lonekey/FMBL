# coding: UTF-8
import torch.nn as nn
import torch.nn.functional as F
# from transformers import RobertaModel
import torch


# class RCModel(nn.Module):
#     def __init__(self, pretrained_path):
#         super(RCModel, self).__init__()
#         self.codebert = RobertaModel.from_pretrained(pretrained_path)
#         self.f1 = nn.Sequential(
#             nn.Linear(768, 2),
#             nn.Softmax(1)
#         )
#
#     def forward(self, sentence):
#         cls = self.codebert(sentence)[1]
#         output = self.f1(cls)
#         return output


class RCModel_CNN(nn.Module):
    def __init__(self, config):
        super(RCModel_CNN, self).__init__()
        self.config = config
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, config.filter_num, (fsz, config.dim)) for fsz in config.filters])
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(1, config.filter_num, (fsz, config.filter_num * 3)) for fsz in config.filters])
        self.convs3 = nn.ModuleList(
            [nn.Conv2d(1, config.filter_num, (fsz, config.dim)) for fsz in config.filters])
        self.f1 = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.filter_num * len(config.filters) * 2, 2),
            nn.Softmax(1)
        )

    def forward(self, report, code):
        # code
        # print(0, code.shape)
        # batch_size*20*20*100
        code = code.view(-1, 1, code.size(2), code.size(3))
        # print(1, code.shape)
        # 3*100-> (batch_size*max_c_l)*filter_num*(max_c_k-3,4,5)*1
        code = [F.relu(conv(code)) for conv in self.convs1]
        # print(2, [c.shape for c in code])
        code = [F.max_pool2d(input=c_item, kernel_size=(c_item.size()[2], 1)) for c_item in code]
        # print(3, [c.shape for c in code])
        code = [x_item.view(x_item.size(0), -1) for x_item in code]
        # print(4, [c.shape for c in code])
        code = torch.cat(code, 1)
        # print(5, code.shape)
        code = code.view(-1, self.config.max_c_l, self.config.filter_num * len(self.config.filters))
        # print(6, code.shape)
        code = code.view(-1, 1, code.size(1), code.size(2))
        # print(7, code.shape)
        # 1024*1*100*300
        code = [F.relu(conv(code)) for conv in self.convs2]
        # print(8, [c.shape for c in code])
        code = [F.max_pool2d(input=c_item, kernel_size=(c_item.size()[2], c_item.size()[3])) for c_item in code]
        # print(9, [c.shape for c in code])
        code = [x_item.view(x_item.size(0), -1) for x_item in code]
        # print(10, [c.shape for c in code])
        code = torch.cat(code, 1)
        # print(11, code.shape)

        # report 1024*100*128
        report = report.view(-1, 1, self.config.maxQueryLength+8, self.config.dim)
        report = [F.relu(conv(report)) for conv in self.convs3]
        # 1024*100*97*1
        report = [F.max_pool2d(input=r_item, kernel_size=(r_item.size()[2], r_item.size()[3])) for r_item in report]
        report = [r_item.view(r_item.size(0), -1) for r_item in report]
        report = torch.cat(report, 1)
        output = torch.cat([code, report], 1)
        # print(12, output.shape)
        prediction = self.f1(output)
        # print(prediction)

        return prediction
