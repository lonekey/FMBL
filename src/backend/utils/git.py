# @Time : 2021/1/12 17:15
# @Author : Cheng Zhu
# @site : https://gitee.com/lonekey
# @File : tool_git.py
import subprocess
import time


def checkout(code_path, commit):
    """
    checkout to commit^
    :param code_path:
    :param commit:
    :return:The result of checkout, True if success, False if error
    """
    command = f"git checkout -f {commit}^"
    p = subprocess.run(command, shell=True, cwd=code_path)
    return p.returncode + 1


def checkout_this(code_path, commit):
    """
    checkout to commit
    :param code_path:
    :param commit:
    :return:The result of checkout, True if success, False if error
    """
    command = f"git checkout -f {commit}"
    p = subprocess.run(command, shell=True, cwd=code_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while p.returncode != 0:
        time.sleep(1)
    # print("success")
    return


def diff(code_path, last_commit, this_commit, mode):
    """

    :param project:
    :param last_commit:
    :param this_commit:
    :param mode: A M D
    :return: list of diff files between commit1 commit2
    """
    command = f"git diff --name-status {last_commit} {this_commit} | findstr \".java\" | findstr \"^{mode}\""
    # print(command)
    p = subprocess.run(command, shell=True, cwd=code_path, stdout=subprocess.PIPE, encoding='utf8')
    recodes = p.stdout.split('\n')[:-1]
    # print(recodes)
    re = []
    for i in range(len(recodes)):
        file = recodes[i].split('\t')[1]
        if not file.startswith('tests'):
            filename = f"{code_path}/{file}"
            re.append(filename)
    return re


def test():
    # print(checkout(PROJECT_LIST[0], "8db6c32"))
    print(diff("E:\\buglocate\\src\\cache\\AspectJ\\code", "b2cd5fa175facc39bd0d1af5a4646b9b39c8bcda", "9319e343d54a65bcfc4a8c19e4305147ce9e27b8", "D"))


if __name__ == "__main__":
    test()
