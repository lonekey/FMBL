import re
stopwords = {"a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually", "after",
             "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along", "already",
             "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow",
             "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate",
             "are", "aren't", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away",
             "awfully", "b", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand",
             "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both",
             "brief", "but", "by", "c", "c'mon", "c's", "came", "can", "can't", "cannot", "cant", "cause", "causes",
             "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning", "consequently",
             "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn't",
             "course", "currently", "d", "definitely", "described", "despite", "did", "didn't", "different", "do",
             "does", "doesn't", "doing", "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg",
             "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever",
             "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f",
             "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly",
             "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go",
             "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly", "has",
             "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here", "here's",
             "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither",
             "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored",
             "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar",
             "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "j", "just",
             "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately", "later", "latter",
             "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely", "little", "look",
             "looking", "looks", "ltd", "m", "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely",
             "might", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "n", "name", "namely", "nd",
             "near", "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine",
             "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere",
             "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only",
             "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside",
             "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed", "please", "plus",
             "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re",
             "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "s",
             "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed",
             "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven",
             "several", "shall", "she", "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow",
             "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified",
             "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell",
             "tends", "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs",
             "them", "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore",
             "therein", "theres", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think",
             "third", "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru",
             "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying",
             "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon",
             "us", "use", "used", "useful", "uses", "using", "usually", "uucp", "v", "value", "various", "very", "via",
             "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll", "we're", "we've",
             "welcome", "well", "went", "were", "weren't", "what", "what's", "whatever", "when", "whence", "whenever",
             "where", "where's", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether",
             "which", "while", "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing",
             "wish", "with", "within", "without", "won't", "wonder", "would", "wouldn't", "x", "y", "yes", "yet", "you",
             "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "z", "zero", "<eow>",
             "<sow>"}
keywords_and_symbols = {"abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const",
                        "continue", "default", "do", "double", "else", "enum", "extends", "final", "finally", "float",
                        "for", "if", "implements", "import", "instanceof", "int", "interface", "long", "native", "new",
                        "package", "private", "protected", "public", "return", "short", "static", "strictfp", "super",
                        "switch", "synchronized", "this", "throw", "throws", "transient", "try", "void", "volatile",
                        "while", "(", ")", ".", ";", ",", "{", "}", "=", "[", "]", ":", "!", "*", "/", "%", "+", "-",
                        ">=", "<=", ">", "<", "==", "!=", "&&", "||", "=", "'", '"', "\r\n", "<<", ">>", "++", "--",
                        "+=", "-=", "*=", "/=", "@", "?", "&", "#", "$", "^", "@", "~"}
format_symbols = {'\r\n', "\"", "\'", "\n", "\t", ";", "//", " +"}
symbols = {"(", ")", ".", ";", ",", "{", "}", "=", "[", "]", ":", "!", "*", "/", "%", "+", "-",
           ">=", "<=", ">", "<", "==", "!=", "&&", "||", "=", "'", '"', "\r\n", "\"", "\'",
           "<<", ">>", "++", "--", "+=", "-=", "*=", "/=", "@", "?", "&", "#", "$", "^", "@", "~", "/", "\\", "|", "\t"}


def camel_case_split(s):
    if "_" in s:
        t = s.lower().split("_")
        return t
    start = 0
    lst = []
    for i in range(1, len(s)):
        if s[i].isupper() and not s[i - 1].isupper():
            lst.append(s[start:i].lower())
            start = i
    lst.append(s[start:].lower())
    return lst


def remove_stopwords(words: list):
    if words is not None:
        words = [i for i in words if i not in stopwords]
    return words


def word_split_and_stop_words_remove(text: str):
    word_list = text.split(' ')
    new_word_list = []
    for word in word_list:
        new_word_list.extend(camel_case_split(word))
    new_word_list = [i for i in new_word_list if i not in stopwords and i not in keywords_and_symbols]
    str_pro = ' '.join(new_word_list)
    return str_pro


def clean_code(text: str):
    # text = re.sub(r"[^A-Za-z(),!?;\'\`]", " ", text)
    text = re.sub(r'\d+', ' ', text)
    for sy in symbols:
        text = text.replace(sy, ' ')
    text = re.sub(' +', ' ', text)
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        words = line.split(' ')
        new_line = ""
        for w in words:
            w_s = " ".join(camel_case_split(w))
            new_line += " " + w_s
        # 去除空行
        if new_line.strip() != '':
            new_lines.extend(new_line.strip().split(' '))
    return new_lines


def clean_str(text, TREC=False):
    # text = re.sub(r"[^A-Za-z(),!?;\'\`]", " ", text)
    text = re.sub(r'\d+', ' ', text)
    for sy in symbols:
        text = text.replace(sy, ' ')
    text = word_split_and_stop_words_remove(text)
    for sy in symbols:
        text = text.replace(sy, ' ')
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)
    text = text.strip() if TREC else text.strip().lower()
    return [text.split(' ')]