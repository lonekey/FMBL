# @Time : 2021/1/5 8:46
# @Author : Cheng Zhu
# @site : https://gitee.com/lonekey
# @File : config.py
# mode
do_train = True
do_eval = False
# model
pretrained_path = "microsoft/codebert-base"
tokenizer_name = "microsoft/codebert-base"
output_dir = "cache"

# train
start_epoch = 0
learning_rate = 0.001
# train_batch_size = 4
num_train_epochs = 100

# dataset
max_r_len = 200
max_c_k = 20
max_c_l = 30
min_vocab_size = 5
dim = 300
negative_file_num = 20
negative_method_num = 100

seed = 233

# CNN
filters = [3, 4, 5]
filter_num = 100
dropout = 0.5
batch_size = 64
