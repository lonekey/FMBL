class Log(object):
    def __init__(self, filename):
        self.filename = filename

    def log(self, *text):
        line = ""
        for item in text:
            line += " " + str(item)
        line += "\n"
        with open(self.filename, 'a+', encoding='utf8') as f:
            f.write(line)
        f.close()
