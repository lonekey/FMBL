import numpy as np
import pickle
from collections import defaultdict
from utils.preprocess import clean_code, clean_str
from tqdm import tqdm
import gensim
from data_model import Project
import random


def load_bin_vec(filename, vocab):
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)  # 模型的导入
    for word in vocab:
        if word in model.vocab:
            word_vecs[word] = model[word]
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    count = 0
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
            count += 1


def getIdxfrom_sent(sent, word_idx_map, code_maxk):
    x = []
    #    pad = filter_h - 1
    #    for i in xrange(pad):
    #        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    if len(x) <= code_maxk:
        while len(x) < code_maxk:
            x.append(0)
    if len(x) >= code_maxk:
        while len(x) > code_maxk:
            x.pop()
    return x


def getIdxfrom_sent_n(sent, max_l, word_idx_map, filter_h=5):
    x = []
    pad = filter_h - 1
    for i in range(pad):
        x.append(0)
    words = sent.split()[:max_l]
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def get_W(word_vecs, k=300):
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    idx_word_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        idx_word_map[i] = word
        i += 1
    return W, word_idx_map, idx_word_map


def alignData(data, code_maxl):
    mm = data.shape[0]
    nn = data.shape[1]
    if mm < code_maxl:
        tt = code_maxl - mm
        aa = np.zeros(nn * tt, dtype="int").reshape(tt, nn)
        new_data = np.vstack((data, aa))
    else:
        new_data = data[:code_maxl, :]
    return new_data


def load_data(_file_path, _code_maxl, _code_maxk, _report_maxl, _test_c, _w2v_file):
    # code_maxl max statements per file
    # code_maxk max words per statement

    vocab = defaultdict(float)
    p: Project = pickle.load(open(_file_path, 'rb'))
    for method in tqdm(p.methods.values()):
        method.content = clean_code(method.content)
        for line in method.content:
            for word in line.split():
                vocab[word] += 1
    for bug in tqdm(p.bugs.values()):
        bug.bug_description = clean_str(bug.bug_description)
        bug.bug_summary = clean_str(bug.bug_summary)
        bug.bug_comments = ' '.join([clean_str(comment['comment']) for comment in bug.bug_comments])
        for text in [bug.bug_description, bug.bug_summary, bug.bug_comments]:
            if len(text) <= 1:
                continue
            words = set(text.split())
            for word in words:
                vocab[word] += 1
    print(len(vocab))
    w2v = load_bin_vec(_w2v_file, vocab)
    add_unknown_words(w2v, vocab)
    W, word_idx_map, idx_word_map = get_W(w2v)
    _train_data, _eval_data = [], []
    # for train
    for _bid, _cid, _report, _code, _label in tqdm(p.getBuggyReportMethodPairs(end=0.9), desc="for train"):
        _report = np.array(getIdxfrom_sent_n(_report, _report_maxl, word_idx_map, filter_h=5))
        _code = np.array([getIdxfrom_sent(i, word_idx_map, _code_maxk) for i in _code])
        if _code.shape[0] == 0:
            continue
        _code = alignData(_code, _code_maxl)
        _train_data.append((_bid, _cid, _report, _code, _label))
    # for eval
    for _bid, _cid, _report, _code, _label in tqdm(p.getBuggyReportMethodPairs(start=0.9, test_num=_test_c), desc="for eval"):
        _report = np.array(getIdxfrom_sent_n(_report, _report_maxl, word_idx_map, filter_h=5))
        _code = np.array([getIdxfrom_sent(i, word_idx_map, _code_maxk) for i in _code])
        if _code.shape[0] == 0:
            continue
        _code = alignData(_code, _code_maxl)
        _eval_data.append((_bid, _cid, _report, _code, _label))
    print(W.shape)
    random.shuffle(_train_data)
    print("Finish loading")
    return _train_data, _eval_data, W, word_idx_map, idx_word_map


if __name__ == "__main__":
    w2v_file = "GoogleNews-vectors-negative300.bin"
    code_maxl = 30  # code_maxl max statements per file
    code_maxk = 20  # code_maxk max words per statement
    report_maxl = 200  # max report length
    test_c = 300  # random choose 300 candidate source code
    file_path = "cache/AspectJ/AspectJ.pkl"
    train_data, eval_data, W, word_idx_map, idx_word_map= load_data(file_path, code_maxl, code_maxk, report_maxl, test_c, w2v_file)
    pickle.dump([train_data, eval_data, W, word_idx_map, idx_word_map], open("cache/AspectJ/parameters.in", "wb"))
    print("Finish processing!")

#     train_data, eval_data, W = pickle.load(open("cache/AspectJ/parameters.in", "rb"))
#     bid, cid, report, code, label = zip(*eval_data)
#     # print(W.shape)
#     # print(report.shape, code.shape)
#     print(len(bid), len(cid), len(label))
