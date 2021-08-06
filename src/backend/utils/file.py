# @Time : 2021/1/12 18:46
# @Author : Cheng Zhu
# @site : https://gitee.com/lonekey
# @File : file.py
import os

def getFileList(code_path):
    # print(code_path)
    file_list = []
    for root, dirs, files in os.walk(code_path):
        for filename in files:
            fullname = f"{root}/{filename}"
            if filename.endswith('.java') and fullname.find(f'{code_path}\\tests') == -1:
            # if filename.endswith('.java'):
                fullname = fullname.replace('\\', '/')
                file_list.append(fullname)
                
    return file_list
