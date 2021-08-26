import numpy as np
import pickle
from collections import defaultdict
from utils.preprocess import clean_code, clean_str
from tqdm import tqdm
import gensim
from data_model import Project
import random
import config


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


def makeVocab(_file_path, _w2v_file):
    vocab = defaultdict(float)
    p: Project = pickle.load(open(_file_path, 'rb'))
    if not hasattr(p, "processed"):
        for method in tqdm(p.methods.values(), desc='methods'):
            method.content = clean_code(method.content)
            for line in method.content:
                for word in line.split():
                    vocab[word] += 1
        for bug in tqdm(p.bugs.values(), desc='bugs'):
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
        p.W = W
        p.word_idx_map = word_idx_map
        p.idx_word_map = idx_word_map
        p.processed = True
        pickle.dump(p, open(_file_path, "wb"))
    return p

def load_data(file_path, config, for_train=False, for_eval=False, for_test=False, for_file=False, for_method=False, w2v_file=None):

    p = makeVocab(file_path, w2v_file)
    _train_data_file, _eval_data_file = [], []
    _train_data_method, _eval_data_method = [], []
    if for_file:
        if for_train:
            for _bid, _cid, _report, _code, _label in tqdm(p.getReportFilePairs(end=0.8, negative_example_num=config.negative_f_num), desc="for train"):
                _report = np.array(getIdxfrom_sent_n(_report, config.max_r_len, p.word_idx_map, filter_h=5))
                _code = np.array([getIdxfrom_sent(i, p.word_idx_map, config.max_c_k) for i in _code])
                if _code.shape[0] == 0:
                    continue
                _code = alignData(_code, config.max_f_l)
                _train_data_file.append((_bid, _cid, _report, _code, _label))
        if for_eval:
            for _bid, _cid, _report, _code, _label in tqdm(p.getReportFilePairs(start=0.8, eval_num=config.test_c), desc="for eval"):
                _report = np.array(getIdxfrom_sent_n(_report, config.max_r_len, p.word_idx_map, filter_h=5))
                _code = np.array([getIdxfrom_sent(i, p.word_idx_map, config.max_c_k) for i in _code])
                if _code.shape[0] == 0:
                    continue
                _code = alignData(_code, config.max_f_l)
                _eval_data_file.append((_bid, _cid, _report, _code, _label))
    if for_method:
        if for_train:
            for _bid, _cid, _report, _code, _label in tqdm(p.getReportMethodPairs(end=0.8, negative_example_num=config.negative_m_num), desc="for train"):
                _report = np.array(getIdxfrom_sent_n(_report, config.max_r_len, p.word_idx_map, filter_h=5))
                _code = np.array([getIdxfrom_sent(i, p.word_idx_map, config.max_c_k) for i in _code])
                if _code.shape[0] == 0:
                    continue
                _code = alignData(_code, config.max_m_l)
                _train_data_method.append((_bid, _cid, _report, _code, _label))
            # for eval
        # _bid, _cid, _report, _code, _label = zip(*_eval_data_file)
        # bug_file_map = dict()
        # for i in range(len(_bid)):
        #     if _bid[i] in bug_file_map.keys():
        #         bug_file_map[_bid[i]].append(_cid[i])
        #     else:
        #         bug_file_map[_bid[i]] = [_cid[i]]
        # print(bug_file_map.keys())
        if for_eval:
            for _bid, _cid, _report, _code, _label in tqdm(p.getReportMethodPairs(start=0.8), desc="for eval"):
                _report = np.array(getIdxfrom_sent_n(_report, config.max_r_len, p.word_idx_map, filter_h=5))
                _code = np.array([getIdxfrom_sent(i, p.word_idx_map, config.max_c_k) for i in _code])
                if _code.shape[0] == 0:
                    continue
                _code = alignData(_code, config.max_m_l)
                _eval_data_method.append((_bid, _cid, _report, _code, _label))
    random.shuffle(_train_data_file)
    random.shuffle(_train_data_method)
    data = {}
    data["file_train"] = _train_data_file
    data["file_eval"] = _eval_data_file
    data["method_train"] = _train_data_method
    data["method_eval"] = _eval_data_method
    print("Finish loading")
    return data, p.W, p.word_idx_map, p.idx_word_map


if __name__ == "__main__":
    PROJECT_LIST = ["AspectJ", "Eclipse_Platform_UI", "Birt", "JDT", "Tomcat"]
    w2v_file = "GoogleNews-vectors-negative300.bin"
    for p in PROJECT_LIST:
        file_path = f"cache/{p}/{p}.pkl"
        data, W, word_idx_map, idx_word_map= load_data(file_path, config, w2v_file=w2v_file, for_train=True, for_file=True, for_eval=True)

        print("Finish processing!")

    # train_data_file, eval_data_file, train_data_method, eval_data_method, W, word_idx_map, idx_word_map = pickle.load(open("cache/AspectJ/parameters.in", "rb"))
    
    # print(W.shape)
    # for data in [train_data_file, eval_data_file, train_data_method, eval_data_method]:
    #     bid, cid, report, code, label = zip(*data)
    #     print(len(bid), len(cid), len(label))
    #     bid, cid, report, code, label = data[0]
    #     print(report.shape, code.shape)


