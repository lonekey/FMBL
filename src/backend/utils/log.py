import datetime


def log(info):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('log.txt', 'a+', encoding='utf8') as f:
        f.write(time+'\t'+info+'\n')
    f.close()
