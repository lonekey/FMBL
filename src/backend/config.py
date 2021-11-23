# @Time : 2021/1/5 8:46
# @Author : Cheng Zhu
# @site : https://gitee.com/lonekey
# @File : config.py
import json


class Config:
    def __init__(self, configfile = None):
        if configfile is None:
            configString = json.load(open('config.json', 'r', encoding='utf-8'))
        else:
            configString = json.load(open(configfile, 'r', encoding='utf-8'))
        # log(configString)
        self.product = configString["product"]
        self.bugRepo = configString["bugRepo"]
        self.gitRepo = configString["gitRepo"]
        self.maxDatasetSize = configString["maxDatasetSize"]
        self.maxQueryLength = configString["maxQueryLength"]
        self.maxCodeK = configString["maxCodeK"]
        self.maxFileLine = configString["maxFileLine"]
        self.maxFuncLine = configString["maxFuncLine"]
        self.useCodeLength = configString["useCodeLength"]
        self.useTFIDF = configString["useTFIDF"]
        self.useLearning = configString["useLearning"]
        self.file = configString["file"]
        self.function = configString["function"]
        # log(configString)
        # # mode
        # do_train = True
        # do_eval = False
        # # model
        # pretrained_path = "microsoft/codebert-base"
        # tokenizer_name = "microsoft/codebert-base"
        self.output_dir = "cache"

        # train
        self.learning_rate = configString["learningRate"]
        self.batch_size = configString["batchSize"]
        self.num_train_epochs = configString["epoch"]

        self.dim = 300
        self.negative_f_num = configString["negFileNum"]
        self.negative_m_num = configString["negMethodNum"]
        self.seed = 233

        # CNN
        self.filters = [3, 4, 5]
        self.filter_num = 100
        self.dropout = 0.5
        self.test_c = 300
    #     self.cs = configString
    #
    # def __repr__(self):
    #     return self.cs

