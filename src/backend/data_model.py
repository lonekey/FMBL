# @Time : 2021/3/15 14:27
# @Author : Cheng Zhu
# @site : https://gitee.com/lonekey
# @File : data_model.py
from typing import Dict
from utils.java_parse import get_method, java_to_json
import uuid


def split_description(filename):
    with open(filename, 'r', errors='ignore') as f:
        lines = f.readlines()
        f.close()
        # 分离code和description
        description = []
        code = []
        for line in lines:
            line = line.lstrip().replace('\n', '')
            if line == '': continue
            if line.find('/*') >= 0:
                description.append(line)
                continue
            if line.startswith('*'):
                description.append(line)
                continue
            if line.find('*/') >= 0:
                description.append(line)
                continue
            if line.startswith('//'):
                description.append(line)
                continue
            else:
                code.append(line)
    return description, code



class Commit:
    def __init__(self, commit_id, commit_time, fixed_bug_id=None):
        self.commit_id = str(commit_id)
        self.commit_time = str(commit_time)
        self.fixed_bug_id = fixed_bug_id
        # 当前提交后的版本文件
        self.current_files = set()
        # 变更了的文件
        self.A = set()
        self.D = set()
        self.M = set()
        # 修改了的方法(A\D\M)
        self.methods = set()

    def __repr__(self):
        # return f"{self.commit_id, self.commit_time, len(self.project_files), [len(i.method_list) for i in
        # self.project_files]}"
        return f"{self.commit_id}"


class Bug:
    def __init__(self, bug_id: str, bug_exist_version, fixed_version, bug_summary, bug_description, bug_comments):
        self.bug_id = str(bug_id)
        self.bug_exist_version = str(bug_exist_version)
        self.fixed_version = [str(fixed_version)]
        self.bug_summary = bug_summary
        self.bug_description = bug_description
        self.bug_comments = bug_comments
        # print(f"bug {bug_id} finished")

    def __repr__(self):
        return f"{self.bug_id}"


class File:
    def __init__(self, project, filename: str, lock=None):
        self.filename = filename
        self.method_list = []
        self.id = uuid.uuid1()
        ml = get_method(filename)
        if ml is None:
            ml = java_to_json(split_description(filename)[1], 0, 0)["method_list"]
        for me in ml:
            method = Method(self.id, me)
            if lock is not None:
                lock.acquire()
            project.methods[method.id] = method
            if lock is not None:
                lock.release()
            self.method_list.append(method.id)
        # print(f"create a file! {file_id} {filename}")

    def __repr__(self):
        # return f"{self.id, self.filename, len(self.method_list), self.create_commit, self.change_history, self.delete_commit}"
        return f"{self.filename}"


class Method:
    # 每次创建新方法时都分配一个key, 认为方法一旦被修改就是全新的方法,方法没有引起变更的提交历史，只能从头创建，一旦创建属性不可修改
    def __init__(self, file_uuid, f: dict):
        self.method_name = f["name"]
        self.content = f["content"]
        self.comment = f["comment"]
        self.start = f["start"]
        self.end = f["end"]
        # 如果这个方法发生修改了，值设置成False
        self.id = uuid.uuid1()
        self.file = file_uuid

    def __repr__(self):
        return f"{self.method_name}"


class Project:
    def __init__(self, project_name):
        self.project_name = project_name
        self.commits: Dict[str, Commit] = {}
        self.bugs: Dict[str, Bug] = {}
        self.files: Dict[uuid.UUID, File]= {}
        self.methods: Dict[uuid.UUID, Method] = {}

    def __repr__(self):
        return f"{self.project_name}"

    def getBugIds(self):
        return [i for i in self.bugs.keys()]

    def getChangedFilesByBugID(self, bugId):
        bug = self.bugs[bugId]
        A = [('A',self.files[file].filename) for i in bug.fixed_version for file in self.commits[i].A]
        D = [('D', self.files[file].filename) for i in bug.fixed_version for file in self.commits[i].D]
        M = [('M', self.files[file].filename) for i in bug.fixed_version for file in self.commits[i].M]
        A.extend(D)
        A.extend(M)
        return A
    
    def getChangedMethodsByBugID(self, bugId):
        bug = self.bugs[bugId]
        methods = [self.methods[method] for i in bug.fixed_version for method in self.commits[i].methods]
        return methods

    def getFilenamesByCommitId(self, commitId):
        filenames = ['/'.join(self.files[i].filename.split('/')[3:]) for i in self.commits[commitId].current_files]
        return filenames

    def getFileIdsByCommitId(self, commitId):
        fileIds = self.commits[commitId].current_files
        return fileIds

    def getFileContentById(self, fileId):
        file = self.files[fileId]
        file.method_list = [self.methods[i] for i in file.method_list]
        return file

    def getLatestCommit(self):
        return list(self.commits.keys())[-1]


if __name__ == "__main__":

    print(uuid.uuid1())

    # description, code = split_description("BrowserManager.java")
    # print(description)
    # print(code)
    # rf = java_to_json(code, 0, 0)
    # fl = rf["method_list"]    # print(fl)