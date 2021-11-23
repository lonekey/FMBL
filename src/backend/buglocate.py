import pickle
from data_model import Project
from utils.file import getFileList
from utils.preprocess import clean_code, clean_str
import time
import process_cv2
from config import Config
import bl_TFIDF
import bl_Length
import numpy as np
import os
from utils.log import log
"""
git log -1 -> commit

"""


def getNewestCommit(product):
    p: Project = pickle.load(open(f'cache/{product}/{product}.pkl', 'rb'))
    newestCommit = list(p.commits.keys())[-1]
    filenames = p.getFilenamesByCommitId(newestCommit)
    return filenames


# def checkFile(product, raw_project_path):
#     raw_path_len = len(raw_project_path.split('\\'))
#     file_list = getFileList(raw_project_path)
#     file_list = ['/'.join(i.split('/')[raw_path_len:]) for i in file_list]
#     repo_file_list = getNewestCommit(product)
#     for f in file_list:
#         if f not in repo_file_list:
#             log(f)
#     log(len(file_list), len(repo_file_list))


def mergeScore(*scores):
    score = np.ones(len(scores[0]))
    for i in scores:
        if len(i) != 0:
            score *= i
    return score


def rank(score, filenames, num, answer=None):
    result = zip(score, filenames)
    result = sorted(result, key=lambda x: x[0], reverse=True)
    if answer is not None:
        for index, i in enumerate(result):
            if i[1] in answer:
                log(f"{index}, {i}")
    return result[:num]


# def predict(product, query):
#     p: Project = pickle.load(open(f'cache/{product}/{product}.pkl', 'rb'))
#
#     cid = p.getLatestCommit()
#     # for bug in p.bugs.values():
#     #     if cid in bug.fixed_version:
#     #         query = bug.bug_summary+'\n'+bug.bug_description
#     #         break
#
#     query = clean_str(open(query).read())
#     fids = p.getFileIdsByCommitId(cid)
#     files = [p.getFileById(i) for i in fids]
#     codes = []
#     filenames = []
#     code_path_len = len(f'cache/{product}/code/')
#     for i in files:
#         filenames.append(i.filename[code_path_len:])
#         code = ""
#         code += i.filename + '\n'
#         for j in i.method_list:
#             j = p.getMethodById(j)
#             code += str(j.content)
#             code += str(j.comment)
#         code = clean_code(code)
#         codes.append(' '.join(code).split(' '))
#
#     score_TFIDF = bl_TFIDF.compute([query.split(' ')], codes)
#     score_Length = bl_Length.compute(codes)
#     score = mergeScore(score_TFIDF, score_Length)
#     # answer= ["cache/AspectJ/code/weaver/src/org/aspectj/weaver/bcel/asm/StackMapAdder.java", "cache/AspectJ/code/org.aspectj.ajdt.core/src/org/aspectj/ajdt/internal/core/builder/AjState.java"]
#     result = rank(score, filenames)
#     return result


def predict_M(config: Config, query):
    # start = time.time()
    score_TFIDF_f, score_Length_f, score_Learning_f, score_TFIDF_m, score_Length_m, score_Learning_m = [], [], [], [], [], []
    p: Project = pickle.load(open(f'{config.output_dir}/{config.product}/{config.product}.pkl', 'rb'))
    # log(p.word_idx_map)
    # end = time.time()
    # log("读取时间", end-start)
    # start = time.time()
    cid = p.getLatestCommit()
    # for bug in p.bugs.values():
    #     if cid in bug.fixed_version:
    #         query = bug.bug_summary+'\n'+bug.bug_description
    #         break
    # print(query)
    query = clean_str(open(query).read())
    # print(query[:100])
    query_idx = process_cv2.getIdxfrom_sent_n(query, config.maxQueryLength, p.word_idx_map, filter_h=5)
    query = [query.split(' ')]
    current_version = f'{config.output_dir}/{config.product}/commit_{cid}.pkl'
    if not os.path.exists(current_version):
        fids = p.getFileIdsByCommitId(cid)
        files = [p.getFileById(i) for i in fids]
        codes_f, codes_m = [], []
        # code_path_len = len(f'cache/{product}/code/')
        mids = []
        for i in files:
            code_f = [clean_str(i.filename)]
            for j in i.method_list:
                mids.append(j)
                j = p.getMethodById(j)
                code_m = str(j.content) + '\n' + str(j.comment) +'\n'
                code_m = clean_code(code_m)
                codes_m.append(code_m)
                code_f.extend(code_m)
            codes_f.append(code_f)
        # !!!
        pickle.dump((list(fids), codes_f, list(mids), codes_m), open(current_version, 'wb'))
    else:
        fids, codes_f, mids, codes_m = pickle.load(open(current_version, 'rb'))

    codes_f_idx, codes_m_idx = [], []
    for file in codes_f:
        file = np.array([process_cv2.getIdxfrom_sent(i, p.word_idx_map, config.maxCodeK) for i in file])
        file = process_cv2.alignData(file, config.maxFileLine)
        codes_f_idx.append(file)
    # codes_f_idx = np.array(codes_f_idx)
    for file in codes_m:
        file = np.array([process_cv2.getIdxfrom_sent(i, p.word_idx_map, config.maxCodeK) for i in file])
        file = process_cv2.alignData(file, config.maxFuncLine)
        codes_m_idx.append(file)
    # codes_m_idx = np.array(codes_m_idx)
    # log(codes_f_idx.shape, len(codes_f_idx), len(fids))
    # log(codes_m_idx.shape, len(codes_m_idx), len(mids))

    codes_f = [' '.join(i).split() for i in codes_f]
    codes_m = [' '.join(i).split() for i in codes_m]
    # print(len(codes_f), len(fids), len(codes_m), len(mids))
    # print(p.getFileById(fids[0]).filename)
    # print(codes_f[0])
    # end = time.time()
    # log("处理时间", end-start)

    # start = time.time()
    # file score
    if config.useTFIDF:
        score_TFIDF_f = bl_TFIDF.compute(query, codes_f)
        score_TFIDF_m = bl_TFIDF.compute(query, codes_m)
    if config.useCodeLength:
        score_Length_f = bl_Length.compute(codes_f)
        score_Length_m = bl_Length.compute(codes_m)
    if config.useLearning:
        import bl_Learning_CNN
        score_Learning_f = bl_Learning_CNN.compute(p.W, "file", config, query_idx, codes_f_idx)
        score_Learning_m = bl_Learning_CNN.compute(p.W, "method", config, query_idx, codes_m_idx)

    # method score

    # end = time.time()
    # log("定位时间", end-start)

    # 排序
    # start = time.time()
    score_f = mergeScore(score_TFIDF_f, score_Learning_f, score_Length_f)
    result_f = rank(score_f, fids, 1000)
    score_m = mergeScore(score_TFIDF_m, score_Learning_m, score_Length_m)
    result_m = rank(score_m, mids, 10000)
    # end = time.time()
    # log("排序时间", end-start)
    # start = time.time()

    # optput
    output={}
    code_path_len = len(f'cache/{config.product}/code/')
    for score, fid in result_f:
        output[fid]={"score": score, "methods": []}
    for score, mid in result_m:
        fid = p.methods[mid].file
        if fid not in output.keys():
            continue
        else:
            output[fid]["methods"].append({"name": p.methods[mid].method_name,"score": score, "start":p.methods[mid].start, "end":p.methods[mid].end})
    output_json = []
    for k, v in output.items():
        v["name"] = p.files[k].filename[code_path_len:]
        output_json.append(v)
    # end = time.time()
    # log("结果输出时间", end-start)
    return output_json[:20]


if __name__ == "__main__":
    start_ti = time.time()
    config = Config('dist/config.json')
    print(config.maxQueryLength)
    output = predict_M(config, 'dist/queryfile.txt')
    print(output)
    end_ti = time.time()
    print(end_ti - start_ti)
