# @Time : 2021/6/30 13:36
# @Author : Cheng Zhu
# @site : https://gitee.com/lonekey
# @File : collect.py
import os
import urllib.request as url
from bs4 import BeautifulSoup
import json
import threading
import time
import pandas as pd
import subprocess
import re
import datetime
from utils.log import log
lock = threading.Lock()


class myThread(threading.Thread):
    def __init__(self, website, save_path, bug_id, bug_link, bug_info_list):
        threading.Thread.__init__(self)
        self.bug_id = bug_id
        self.bug_link = bug_link
        self.save_path = save_path
        self.website = website
        self.bug_info_list = bug_info_list

    def run(self):
        try:
            if not os.path.exists(f"{self.save_path}/{self.bug_id}.html"):
                request = f"{self.website}/{self.bug_link}"
                content = url.urlopen(request).read()
                open(f'{self.save_path}/{self.bug_id}.html', 'wb').write(content)
            else:
                content = open(f"{self.save_path}/{self.bug_id}.html", 'rb')
            bug_info = {"id": self.bug_id}
            summary, report_time, modified_time, comments = get_bug_info(content)
            bug_info["summary"] = summary
            bug_info["description"] = comments[0]
            bug_info["comments"] = comments[1:]
            bug_info["report_time"] = report_time
            bug_info["modified_time"] = modified_time
            lock.acquire()
            self.bug_info_list.append(bug_info)
            lock.release()
        except Exception:
            log(f"error")


def get_bug_info(content):
    soup = BeautifulSoup(content, 'html.parser')
    col2 = soup.find('td', {"id": "bz_show_bug_column_2"}).table.select('tr')
    report_time = [s for s in col2[0].td.strings][0].replace('by', '').strip()
    modified_time = [s for s in col2[1].td.strings][0].replace('(', '').strip()
    summary = soup.find("span", {"id": "short_desc_nonedit_display"}).string
    comments = soup.find("table", {"class": "bz_comment_table"}).tr.td.select('div')
    comment_list = []
    for comment in comments[::2]:
        c_item = {"user": comment.find("span", {"class": "fn"}).string,
                  "time": comment.find("span", {"class": "bz_comment_time"}).string.replace('\n', '').strip(),
                  "comment": ' '.join(comment.pre.strings)}
        comment_list.append(c_item)
    return summary, report_time, modified_time, comment_list


def getBugInfoList(product):
    with open(f"bug_info/{product}.json", 'r') as f:
        bug_info_list = json.load(f)
        f.close()
    return bug_info_list


def get_bug_links(content):
    soup = BeautifulSoup(content, 'html.parser')
    bugs = [child for child in soup.table.children if child != "\n"]
    bugs_id = [b.td.a.string for b in bugs[1:]]
    bugs_link = [b.td.a['href'] for b in bugs[1:]]
    return dict(zip(bugs_id, bugs_link))


def collect_bug_report(product, website, component=None):
    """ Fetch issues that match given jql query """
    if component:
        save_path = f"cache/{product}/{component}"
    else:
        save_path = f"cache/{product}"
    issues_path = f"{save_path}/issues"
    os.makedirs(issues_path, exist_ok=True)
    if not os.path.exists(f"{issues_path}/total.html"):
        request = f"{website}/buglist.cgi?bug_status=RESOLVED&limit=0&product={product}&query_format=advanced&resolution=FIXED"
        if component:
            request+= f"&component={component}"
        content = url.urlopen(request).read()
        open(f"{issues_path}/total.html", 'wb').write(content)
    else:
        content = open(f"{issues_path}/total.html")
    bug_links = get_bug_links(content)
    bug_info_list = []
    threads = []
    for bug_id, bug_link in bug_links.items():
        thread = myThread(website, issues_path, bug_id, bug_link, bug_info_list)
        threads.append(thread)
        thread.start()
    while len(threads) != 0:
        if not threads[-1].is_alive():
            threads.pop()
        else:
            time.sleep(1)
    with open(f"{save_path}/bug_report.json", 'w') as f:
        json.dump(bug_info_list, f)
        f.close()
    


def getTimeStamp(date):
    result = re.search(r"[\-\+]\d+", date)
    if result:
        time_area = result.group()
        symbol = time_area[0]
        offset = int(time_area[1]) + int(time_area[2])
        if symbol == "+":
            format_str = '%a %b %d %H:%M:%S %Y ' + time_area
            if "UTC" in date:
                format_str = '%a, %d %b %Y %H:%M:%S ' + time_area + ' (UTC)'
            if "GMT" in date:
                format_str = '%a, %d %b %Y %H:%M:%S ' + time_area + ' (GMT)'
            if "CST" in date:
                format_str = '%a, %d %b %Y %H:%M:%S ' + time_area + ' (CST)'
            utcdatetime = time.strptime(date, format_str)
            tempsTime = time.mktime(utcdatetime)
            tempsTime = datetime.datetime.fromtimestamp(tempsTime)
            if offset > 8:
                offset = offset - 8
            tempsTime = tempsTime + datetime.timedelta(hours=offset)
            localtimestamp = tempsTime.strftime("%Y-%m-%d %H:%M:%S")
        else:
            format_str = '%a %b %d %H:%M:%S %Y ' + time_area
            utcdatetime = time.strptime(date, format_str)
            tempsTime = time.mktime(utcdatetime)
            tempsTime = datetime.datetime.fromtimestamp(tempsTime)
            tempsTime = tempsTime + datetime.timedelta(hours=(offset + 8))
            localtimestamp = tempsTime.strftime("%Y-%m-%d %H:%M:%S")
    return localtimestamp


def collect_git_log(product, productGitPath):
    command = "git log"
    p1 = subprocess.run(command, shell=True, cwd=productGitPath, stdout=subprocess.PIPE, encoding='ISO-8859-1')
    recode = p1.stdout
    open('recode.txt', 'w', encoding='utf-8').write(recode)
    recode = recode.split('\n')
    data = recode
    commit_log = []
    tmp = {"description": ""}
    for i in data:
        if i.find("commit") != -1:
            if 'commit' in tmp.keys():
                commit_log.append(tmp)
            # log(tmp["descripton"])
            tmp = {"description": ""}
            tmp["commit"] = i.split(' ')[1]
        if i.find("Author") != -1:
            tmp["Author"] = ' '.join(i.split(' ')[1:])
        if i.find("Date") != -1:
            date = ' '.join(i.split(' ')[3:])
            try:
                tmp["Date"] = getTimeStamp(date)
            except:
                continue
        if i.startswith(' '):
            if "descripton" not in tmp.keys():
                tmp["description"] = i
            else:
                tmp["description"] += i
    df = pd.DataFrame(commit_log, columns=["commit", "Author", "Date", "description"])
    df.to_csv(f"cache/{product}/git_log.csv", index=None)


def matchRC(product):
    bug_report_path = f"cache/{product}/bug_report.json"
    git_log_path = f'cache/{product}/git_log.csv'
    br = json.load(open(bug_report_path, encoding='utf-8'))
    gl = pd.read_csv(open(git_log_path, encoding='utf-8'))
    bug_repo = {}
    for item in br:
        fixCommit = {}
        for i in gl.itertuples():
            if str(i.description).find(str(item['id'])) != -1:
                if type(i.commit) is str:
                    if str(i.description).find('test') != -1 and str(i.description).find('fix') == -1:
                        pass
                    else:
                        fixCommit[i.commit]= i.Date
        if len(fixCommit.keys()) > 0:
            item['fixCommit'] = fixCommit
            bug_repo[item['id']] = item
    json.dump(bug_repo, open(f'cache/{product}/bug_repo.json', 'w', encoding='utf-8'))
    # for i in bug_repo:
    #     log(i['fixCommit'])
        


# collect_git_log('AspectJ', 'E:\SBL\Repository\org.aspectj')
# match('AspectJ')
# bug_repo = json.load(open('cache/AspectJ/bug_repo.json', encoding='utf-8'))
# for i in bug_repo:
#     log(i['fixCommit'])

# collect_bug_report('Platform', 'https://bugs.eclipse.org/bugs')

def getBugID():
    website = "https://bz.apache.org/"
    for filename in os.listdir('cache/exp_data'):
        log(filename)
        save_path = f'cache/{filename[:-4]}'
        issues_path = f'cache/{filename[:-4]}/issues'
        os.makedirs(issues_path, exist_ok=True)
        with open(f"cache/exp_data/{filename}", 'r') as f:
            soup = BeautifulSoup(f, "lxml")
        f.close()
        lst = soup.find("database").select("table")[::-1]
        bug_ids = [str(bug_info.select('[name=bug_id]')[0].string) for bug_info in lst]

        # bug_report_path = f"{save_path}/bug_report.json"
        # git_log_path = f'{save_path}/git_log.csv'
        # br = json.load(open(bug_report_path, encoding='utf-8'))
        # gl = pd.read_csv(open(git_log_path, encoding='utf-8'))
        # bug_commit_map = {}
        # for bug_info in lst:
        #     bug_id = str(bug_info.select('[name=bug_id]')[0].string)
        #     commitid= str(bug_info.select('[name=commit]')[0].string)
        #     bug_commit_map[bug_id] = commitid
        # bug_repo = {}
        # for item in br:
        #     fixCommit = {}
        #     for i in gl.itertuples():
        #         if str(i.commit).startswith(bug_commit_map[item['id']]):
        #             if type(i.commit) is str:
        #                 fixCommit[i.commit]= i.Date
        #     if len(fixCommit.keys()) > 0:
        #         item['fixCommit'] = fixCommit
        #         bug_repo[item['id']] = item
        # json.dump(bug_repo, open(f'{save_path}/bug_repo.json', 'w', encoding='utf-8'))

        bug_repos = json.load(open(f'cache/{filename[:-4]}/bug_repo.json', 'r'))
        bug_reports = json.load(open(f'cache/{filename[:-4]}/bug_report.json', 'r'))
        log(len(bug_ids), len(bug_reports), len(bug_repos))
        # bug_links = {}
        # for bug_id in bug_ids:
        #     bug_links[bug_id] = "show_bug.cgi?id="+bug_id
        # bug_info_list = []
        # threads = []
        # for bug_id, bug_link in bug_links.items():
        #     thread = myThread(website, issues_path, bug_id, bug_link, bug_info_list)
        #     threads.append(thread)
        #     thread.start()
        # while len(threads) != 0:
        #     if not threads[-1].is_alive():
        #         threads.pop()
        #     else:
        #         time.sleep(1)
        # with open(f"{save_path}/bug_report.json", 'w') as f:
        #     json.dump(bug_info_list, f)
        #     f.close()
        # return
# getBugID()
# collect_git_log('Tomcat', 'E:\\buglocate\src\\backend\\cache\\Tomcat\\code')
# matchRC('Tomcat')