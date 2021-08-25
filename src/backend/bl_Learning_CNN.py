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
import os
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
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


def train(model, train_data, eval_data, W, device, level, learning_rate, batch_size, num_train_epochs):
    best_MRR = 0
    Loss = nn.CrossEntropyLoss(weight=torch.tensor([1, config.cost], dtype=torch.float32).to(device))
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    t_loss = []
    batch_iter = DatasetIterater(train_data, batch_size)
    for epoch in range(num_train_epochs):
        batch_acc, batch_loss, batch_t_p, batch_t_n = [], [], [], []
        for bid, cid, report, code, label in tqdm(batch_iter, desc="train iter"):
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
        avg_loss, accuracy, t_p, t_n, MRR, top_1, top_5, top_10 = evaluate(model, eval_data, W, device, batch_size)
        print(f"MRR: {MRR} top_1: {top_1} top_5: {top_5} top_10: {top_10}\neval loss: {avg_loss} eval_accuracy: {accuracy} t_p: {t_p} t_n: {t_n}")
        scheduler.step()
        if best_MRR < MRR:
            best_MRR = MRR
            checkpoint_prefix = f'AspectJ/{level}'
            output_dir = os.path.join(config.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            print("Saving model checkpoint to %s", output_dir)



def evaluate(model, eval_data, W, device, batch_size):
    t_loss, bids, cids, scores, results, labels = [], [], [], [], [], []
    Loss = nn.CrossEntropyLoss()
    batch_iter = DatasetIterater(eval_data, batch_size)
    for bid, cid, report, code, label in tqdm(batch_iter, desc="eval iter"):
        labels.extend(label)
        bids.extend(bid)
        cids.extend(cid)
        report, code, label = torch.tensor(report, dtype=torch.long), torch.tensor(code,
                                                                                   dtype=torch.long), torch.tensor(
            label, dtype=torch.long)
        report, code, label = report.to(device), code.to(device), label.to(device)
        code = W[code]
        report = W[report]
        model.eval()
        prediction = model(report, code)  # 预测分数
        score = torch.transpose(prediction, 0, 1)[1].cpu().detach().numpy()
        scores.extend(score)
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
#     print(len(bids), len(cids), len(scores), len(labels))
    MRR, top_1, top_5, top_10 = compute_tric(bids, cids, scores, labels)
    return avg_loss, accuracy, t_p, t_n, MRR, top_1, top_5, top_10


def compute_tric(bids, cids, scores, labels):
    MAP, MRR, count_1, count_5, count_10 = 0, 0, 0, 0, 0
    result = {}
    for i in range(len(bids)):
        if bids[i] not in result.keys():
            result[bids[i]]=[(cids[i], scores[i], labels[i])]
        else:
            result[bids[i]].append((cids[i], scores[i], labels[i]))
    bug_num = len(result.keys())
    for k, v in result.items():
        v.sort(key=lambda x: x[1], reverse=True)
        for index, item in enumerate(v):
            if item[2] == 1:
                RR = 1/(index+1)
                MRR += RR
                if index < 10:
                    count_10 += 1
                    if index < 5:
                        count_5 += 1
                        if index < 1:
                            count_1 += 1
                break
    MRR = MRR / bug_num
    top_1 = count_1 / bug_num
    top_5 = count_5 / bug_num
    top_10 = count_10 / bug_num
    return MRR, top_1, top_5, top_10
        

def start_train(level):
    """
    load data and then sent data to train, save the final model
    :param train_from_start: use the initial model if True , else use pretrained model in "start_epoch-1"
    :return: None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data_file, eval_data_file, train_data_method, eval_data_method, W, _, _ = pickle.load(open("cache/AspectJ/parameters.in", "rb"))
    if level == "file":
        config.max_c_l = config.max_f_l
        train_data = train_data_file
        eval_data = eval_data_file
    if level == "method":
        config.max_c_l = config.max_m_l
        train_data = train_data_method
        eval_data = eval_data_method
    W = torch.tensor(W, dtype=torch.float32)
    W = W.to(device)
    model = RCModel_CNN(config)
     # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
    model.to(device)
    print("model loaded")
    train(model, train_data, eval_data, W, device, level, config.learning_rate, config.batch_size, config.num_train_epochs)


def start_evaluate(level):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_data_file, _, eval_data_method, W, _, _ = pickle.load(open("cache/AspectJ/parameters.in", "rb"))
    if level == "file":
        eval_data = eval_data_file
        config.max_c_l = config.max_f_l
    if level == "method":
        config.max_c_l = config.max_m_l
        eval_data = eval_data_method
    W = torch.tensor(W, dtype=torch.float32)
    W = W.to(device)
    model = RCModel_CNN(config)
    checkpoint_prefix = f'AspectJ/{level}'
    output_dir = os.path.join(config.output_dir, '{}'.format(checkpoint_prefix))
    output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
    model.load_state_dict(torch.load(output_dir))
    model.to(device)
    print("model loaded")
    avg_loss, accuracy, t_p, t_n, MRR, top_1, top_5, top_10 = evaluate(model, eval_data, W, device, config.batch_size)
    print(MRR, top_1, top_5, top_10)
if __name__ == "__main__":
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    file = 'file'
    method = 'method'
    start_train(file)
    start_evaluate(file)
    start_train(method)
    start_evaluate(method)
