from abc import ABC

import torch
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
import config
from models import RCModel
import os
import pickle
from data_model import Project
from torch.nn import CrossEntropyLoss
import random
import multiprocessing
from tqdm import tqdm
import numpy as np


def main():
    cpu_count = 1
    pool = multiprocessing.Pool(cpu_count)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(config.pretrained_path)
    model = RCModel(config.pretrained_path)
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
    train(model, tokenizer, device, pool)


def train(model, tokenizer, device, pool):
    train_dataset = myDataset('cache/AspectJ/AspectJ.pkl', tokenizer, pool).examples
    random.shuffle(train_dataset)
    dataIter = DatasetIterater(train_dataset, config.train_batch_size)
    # get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(dataIter) * config.num_train_epochs)
    model.zero_grad()
    model.train()
    tr_num, tr_loss, best_mrr = 0, 0, 0
    predicts, labels = [], []
    print("train start...")
    for idx in range(config.num_train_epochs):
        for step, batch in enumerate(dataIter):
            # get inputs
            bid, cid, tokens_ids, label = batch
            labels.extend(label)
            sentence = torch.tensor(tokens_ids).to(device)
            label = torch.tensor(label).to(device)
#             print(sentence, label)
            output = model(sentence)
#             print(output)
            # print(label)

            # calculate scores and loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(output, label)
            out = torch.argmax(output, 1).cpu().numpy()
            predicts.extend(out)

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % 100 == 0:
                # print(predicts)
                # print(labels)
                count = 0
                for i, v in enumerate(predicts):
                    if v == labels[i]:
                        count += 1
                print("epoch {} step {} loss {} acc {}".format(idx, step + 1, round(tr_loss / tr_num, 5), count/len(labels)))
                tr_loss = 0
                tr_num = 0
                predicts, labels = [], []

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # # evaluate
        # results = evaluate(args, model, tokenizer, args.eval_data_file, pool, eval_when_training=True)
        # for key, value in results.items():
        #     logger.info("  %s = %s", key, round(value, 4))
        #
        # # save best model
        # if results['eval_mrr'] > best_mrr:
        #     best_mrr = results['eval_mrr']
        #     logger.info("  " + "*" * 20)
        #     logger.info("  Best mrr:%s", round(best_mrr, 4))
        #     logger.info("  " + "*" * 20)
        #
        #     checkpoint_prefix = 'checkpoint-best-mrr'
        #     output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        #     if not os.path.exists(output_dir):
        #         os.makedirs(output_dir)
        #     model_to_save = model.module if hasattr(model, 'module') else model
        #     output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
        #     torch.save(model_to_save.state_dict(), output_dir)
        #     logger.info("Saving model checkpoint to %s", output_dir)





class myDataset:

    def __init__(self, file_path, tokenizer, pool):
        """

        :param file_path:
        :param tokenizer:
        """
        file_path = file_path.replace('\\', '/')
        prefix = file_path.split('/')[-1][:-4]
        cache_file = config.output_dir + '/' + prefix + '.pkl'
        print(cache_file)
        if os.path.exists(cache_file):
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            self.examples = []
            p: Project = pickle.load(open(file_path, 'rb'))
            print(len(p.files), len(p.methods), len(p.commits), len(p.bugs))
            # for i in p.getBuggyReportFilePairs():
            #     print(i[1])
            #     break
            data = [(i, tokenizer) for i in p.getBuggyReportMethodPairs()]
            self.examples = pool.map(tokenize, tqdm(data, total=len(data)))
            pickle.dump(self.examples, open(cache_file, 'wb'))
        print("dataset loaded", len(self.examples))

    def __len__(self):
        return len(self.examples)


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


def tokenize(item):
    data, tokenizer = item
    bid, cid, report, code, label = data
    nl_tokens = tokenizer.tokenize(report)
    code_tokens = tokenizer.tokenize(code)
    tokens = [tokenizer.cls_token] + nl_tokens[:254] + [tokenizer.sep_token] + code_tokens[:254] + [tokenizer.sep_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = 512 - len(tokens_ids)
    tokens_ids += [tokenizer.pad_token_id] * padding_length
    return bid, cid, tokens_ids, label


if __name__ == "__main__":
    main()











