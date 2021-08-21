# @Time : 2021/1/15 16:58
# @Author : Cheng Zhu
# @site : https://gitee.com/lonekey
# @File : java_parse.py
import javalang as jl


def __get_start_end_for_node(tree, node_to_find):
    start = None
    end = None
    for path, node in tree:
        if start is not None and node_to_find not in path:
            end = node.position
            return start, end
        if start is None and node == node_to_find:
            start = node.position
    return start, end


def __get_string(data, start, end):
    if start is None:
        return ""

    # positions are all offset by 1. e.g. first line -> lines[0], start.line = 1
    end_pos = None

    if end is not None:
        end_pos = end.line - 1

    lines = data.splitlines(True)
    string = "".join(lines[start.line - 1:end_pos])

    # When the method is the last one, it will contain a additional brace
    # if end is None:
    # left = string.count("{")
    # right = string.count("}")
    # if right - left == 1:
    p = string.rfind("}")
    string = string[:p+1]

    return string


def get_method(filename):
    data = open(filename, errors='ignore').read()
    try:
        tree = jl.parse.parse(data)
        # print(list(tree))
    except jl.parser.JavaSyntaxError:
        # print("JavaSyntaxError", filename)
        return None
    except jl.tokenizer.LexerError:
        return None
    methods = []
    for _, node in tree.filter(jl.tree.MethodDeclaration):
        method = {}
    # if isinstance(node, jl.tree.AnnotationMethod):

    # if isinstance(node, jl.tree.MethodDeclaration):
        start, end = __get_start_end_for_node(tree, node)
        method["name"] = node.name
        method["content"] = __get_string(data, start, end)
        method["comment"] = node.documentation
        method["start"] = start
        method["end"] = end
        methods.append(method)
    return methods


def java_to_json(code, method_id, pre_start):
    file = {}
    count = []
    item = 0
    file2 = ""
    for l in code:
        file2 = file2 + l+"\n"
        count.append(len(file2))
    start = []
    end = []
    method_list = []
    for i in range(len(file2)):
        if file2[i] == "{":
            start.append(i)
        if file2[i] == "}":
            end.append(i)
        if len(start) == len(end) + 1 and len(end) > 0:
            a, b, c = 0, 0, 0
            # { 位置  } 位置 方法起始位置
            method_name = ''
            while len(start) > 1:
                a = start.pop()
            b = end.pop()

            for j in range(len(count)):
                # j : "{" 所在的那一行
                if count[j] >= a:
                    if j == 0:
                        c = 0
                    else:
                        c = count[j - 1]
                        # 含有方法名的一行
                        method_name = code[j]
                        # 处理内部类
                        if method_name.find('class') != -1:
                            # print(file2[c:b + 1])
                            ml = java_to_json(file2[c:b + 1].split('\n'), method_id, c)["method_list"]
                            method_list.extend(ml.copy())
                            method_id += len(ml)
                        elif method_name.find('(') == -1:
                            c = count[j - 2]
                            # 含有方法名的一行
                            method_name = code[j-1]
                    break

            end = []
            if method_name.find('class') == -1:
                res = ""
                for k in method_name:
                    if k != '(':
                        res += k
                    else:
                        break
                method = {}
                method_name = res.strip().split(' ')[-1]
                # print(method_name)
                method["id"] = method_id
                method["name"] = method_name
                method_id += 1
                method["content"] = file2[c:b+1]
                method["start"] = pre_start+c
                method["end"] = pre_start+b+1
                method["comment"] = None
                if not method["content"].startswith("\n\t"):
                    method_list.append(method.copy())
                else:
                    method_id -= 1

    file["method_list"] = method_list
    return file


