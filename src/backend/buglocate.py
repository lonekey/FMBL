import pickle
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


def checkFile(product, raw_project_path):
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


def rank(score, filenames, num, answer=None):
    result = zip(score, filenames)
    result = sorted(result, key=lambda x: x[0], reverse=True)
    if answer is not None:
        for index, i in enumerate(result):
            if i[1] in answer:
                print(index, i)
    return result[:num]


def predict(product, query):
    p: Project = pickle.load(open(f'cache/{product}/{product}.pkl', 'rb'))
    cid = p.getLatestCommit()
    # for bug in p.bugs.values():
    #     if cid in bug.fixed_version:
    #         query = bug.bug_summary+'\n'+bug.bug_description
    #         break

    query = clean_str(open(query).read())
    fids = p.getFileIdsByCommitId(cid)
    files = [p.getFileById(i) for i in fids]
    codes = []
    filenames = []
    code_path_len = len(f'cache/{product}/code/')
    for i in files:
        filenames.append(i.filename[code_path_len:])
        code = ""
        code += i.filename + '\n'
        for j in i.method_list:
            j = p.getMethodById(j)
            code += str(j.content)
            code += str(j.comment)
        code = clean_code(code)
        codes.append(' '.join(code).split(' '))
    
    score_TFIDF = bl_TFIDF.compute([query.split(' ')], codes)
    score_Length = bl_Length.compute(codes)
    score = mergeScore(score_TFIDF, score_Length)
    # answer= ["cache/AspectJ/code/weaver/src/org/aspectj/weaver/bcel/asm/StackMapAdder.java", "cache/AspectJ/code/org.aspectj.ajdt.core/src/org/aspectj/ajdt/internal/core/builder/AjState.java"]
    result = rank(score, filenames)
    return result


def predict_M(product, query):
    p: Project = pickle.load(open(f'cache/{product}/{product}.pkl', 'rb'))
    cid = p.getLatestCommit()
    # for bug in p.bugs.values():
    #     if cid in bug.fixed_version:
    #         query = bug.bug_summary+'\n'+bug.bug_description
    #         break

    query = clean_str(open(query).read())
    fids = p.getFileIdsByCommitId(cid)
    files = [p.getFileById(i) for i in fids]
    codes_f, codes_m = [], []
    # code_path_len = len(f'cache/{product}/code/')
    mids = []
    for i in files:
        code_f = ""
        code_f += i.filename + '\n'
        for j in i.method_list:
            mids.append(j)
            code_m = ""
            j = p.getMethodById(j)
            code_f += str(j.content)
            code_f += str(j.comment)
            code_m += str(j.content)
            code_m += str(j.comment)
            code_m = clean_code(code_m)
            codes_m.append(' '.join(code_m).split(' '))
        code_f = clean_code(code_f)
        codes_f.append(' '.join(code_f).split(' '))
    # file score
    score_TFIDF_f = bl_TFIDF.compute([query.split(' ')], codes_f)
    score_Length_f = bl_Length.compute(codes_f)
    score_f = mergeScore(score_TFIDF_f, score_Length_f)
    result_f = rank(score_f, fids, 100)

    # method score
    score_TFIDF_m = bl_TFIDF.compute([query.split(' ')], codes_m)
    score_Length_m = bl_Length.compute(codes_m)
    score_m = mergeScore(score_TFIDF_m, score_Length_m)
    result_m = rank(score_m, mids, 1000)

    # optput
    output={}
    code_path_len = len(f'cache/{product}/code/')
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
    return output_json[:20]




if __name__ == "__main__":
    start = time.time()
    output = predict_M('AspectJ', 'queryfile.txt')
    print(output)
    end = time.time()
    print(end - start)
