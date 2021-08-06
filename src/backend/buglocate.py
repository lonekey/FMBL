
import pickle
import re
from data_model import Project
from utils.file import getFileList
from utils.preprocess import clean_code, clean_str
import time
import bl_TFIDF
import bl_Length
import numpy as np
"""
git log -1 -> commit

"""
def getNewestCommit(product):
    p: Project = pickle.load(open(f'cache/{product}/{product}.pkl', 'rb'))
    newestCommit = list(p.commits.keys())[-1]
    filenames = p.getFilenamesByCommitId(newestCommit)
    return filenames

def checkfile(product, raw_project_path):
    raw_path_len = len(raw_project_path.split('\\'))
    file_list = getFileList(raw_project_path)
    file_list = ['/'.join(i.split('/')[raw_path_len:]) for i in file_list]
    repo_file_list = getNewestCommit(product)
    for f in file_list:
        if f not in repo_file_list:
            print(f)
    print(len(file_list), len(repo_file_list))

def mergeScore(*scores):
    score = np.ones(len(scores[0]))
    for i in scores:
        score *= i
    return score


def rank(score, filenames, answer=None):
    result = zip(score, filenames)
    result = sorted(result, key= lambda x: x[0], reverse=True)
    if answer is not None:
        for index, i in enumerate(result):
            if i[1] in answer:
                print(index, i)
    return result[:100]

def predict(product, query):
    p: Project = pickle.load(open(f'cache/{product}/{product}.pkl', 'rb'))
    cid = p.getLatestCommit()
    # print(cid)
    # for bug in p.bugs.values():
    #     if cid in bug.fixed_version:
    #         query = bug.bug_summary+'\n'+bug.bug_description
    #         break

    query = clean_str(open(query).read())
    fids = p.getFileIdsByCommitId(cid)
    files = [p.getFileContentById(i) for i in fids]
    codes = []
    filenames = []
    code_path_len = len(f'cache/{product}/code/')
    for i in files:
        filenames.append(i.filename[code_path_len:])
        code = ""
        code+= i.filename+'\n'
        for j in i.method_list:
            code += str(j.content)
            code += str(j.comment)
        code = clean_code(code)
        codes.append(code)
    score_TFIDF = bl_TFIDF.compute(query, codes)
    score_Length = bl_Length.compute(codes)
    score = mergeScore(score_TFIDF, score_Length)
    # answer= ["cache/AspectJ/code/weaver/src/org/aspectj/weaver/bcel/asm/StackMapAdder.java", "cache/AspectJ/code/org.aspectj.ajdt.core/src/org/aspectj/ajdt/internal/core/builder/AjState.java"]
    result = rank(score, filenames)
    return result

if __name__ == "__main__":
    start = time.time()
    predict('AspectJ', '123')
    end = time.time()
    print(end-start)