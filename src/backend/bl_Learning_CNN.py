# @Time : 2021/4/5 16:31
# @Author : Cheng Zhu
# @site : https://gitee.com/lonekey
# @File : main.py

import pickle
from torch import nn
import numpy as np
from torch import optim
from models import RCModel_CNN
import config
import torch


class DatasetIterater:
    def __init__(self, batches, batch_size):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            return zip(*batches)

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            return zip(*batches)

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def train(model, train_data, eval_data, W, device, learning_rate, batch_size, num_train_epochs):
    Loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    t_loss = []
    batch_iter = DatasetIterater(train_data, batch_size)
    for epoch in range(num_train_epochs):
        batch_acc, batch_loss, batch_t_p, batch_t_n = [], [], [], []
        for bid, cid, report, code, label in batch_iter:
            report, code, label = torch.tensor(report, dtype=torch.long), torch.tensor(code,
                                                                                       dtype=torch.long), torch.tensor(
                label, dtype=torch.long)
            report, code, label = report.to(device), code.to(device), label.to(device)
            code = W[code]
            report = W[report]
            model.train()
            prediction = model(report, code)  # 预测分数
            p = []  # 正确预测的 t_p + t_n
            loss = Loss(prediction, label)
            t_loss.append(loss.item())
            batch_loss.append(loss.item())
            out = torch.argmax(prediction, 1)  # 预测结果
            for i in range(len(out)):
                if out[i] == label[i]:
                    p.append(i)
            accuracy = len(p) / len(out)
            batch_acc.append(accuracy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_info = f"epoch_train: {epoch} \tloss: {sum(batch_loss) / len(batch_loss)}\tacc: {sum(batch_acc) / len(batch_acc)} "
        print(train_info)
        eval_loss, eval_accuracy, t_p, t_n = evaluate(model, eval_data, W, device, batch_size)
        print(f"eval loss: {eval_loss} eval_accuracy: {eval_accuracy} t_p: {t_p} t_n: {t_n}")
        scheduler.step()


def evaluate(model, eval_data, W, device, batch_size):
    t_loss, results, labels = [], [], []
    Loss = nn.CrossEntropyLoss()
    batch_iter = DatasetIterater(eval_data, batch_size)
    for bid, cid, report, code, label in batch_iter:
        labels.extend(label)
        report, code, label = torch.tensor(report, dtype=torch.long), torch.tensor(code,
                                                                                   dtype=torch.long), torch.tensor(
            label, dtype=torch.long)
        report, code, label = report.to(device), code.to(device), label.to(device)
        code = W[code]
        report = W[report]
        model.eval()
        prediction = model(report, code)  # 预测分数
        out = torch.argmax(prediction, 1)  # 预测结果
        results.extend(out)
        loss = Loss(prediction, label)
        t_loss.append(loss.item())
    avg_loss = sum(t_loss) / len(t_loss)
    p_count, t_p, t_n = 0, 0, 0
    p, n = 0, 0
    for i in range(len(results)):
        if labels[i] == 1:
            p += 1
        else:
            n += 1
        if results[i] == labels[i]:
            p_count += 1
            if results[i] == 1:
                t_p += 1
            else:
                t_n += 1
    accuracy = p_count / len(labels)
    print(f"{t_p}/{p} {t_n}/{n}")
    t_p = t_p / p  # 正样本预测正确的比率
    t_n = t_n / n  # 负样本预测正确的比率
    return avg_loss, accuracy, t_p, t_n


def start_train(train_from_start=True):
    """
    load data and then sent data to train, save the final model
    :param train_from_start: use the initial model if True , else use pretrained model in "start_epoch-1"
    :return: None
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data, eval_data, W = pickle.load(open("cache/AspectJ/parameters.in", "rb"))
    W = torch.tensor(W, dtype=torch.float32)
    W = W.to(device)
    model = RCModel_CNN(config)
    # if not train_from_start:
    #     model_path = save_path + f"{project}_{locate_level}_{dim}_{index}_{start_epoch - 1}.model"
    #     print(f"load model from {model_path}")
    #     model.load_state_dict(torch.load(model_path))
    model.to(device)
    print("model loaded")
    train(model, train_data, eval_data, W, device, config.learning_rate, config.batch_size, config.num_train_epochs)


if __name__ == "__main__":
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    start_train()
