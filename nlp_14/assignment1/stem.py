#coding:utf-8
import re
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

class Stem(object):
    def __init__(self):
        self.dic = set()
        with open('data/dic_ec.txt') as f:
            for line in f:
                line = line
                word = line.split('')[0]
                self.dic.add(word)
        self.rules = list()
        with open('data/rules.txt') as f:
            for line in f:
                line = line.strip()
                line = line.replace(' ','')
                self.rules.append(line.split('->'))

    def stem(self,word):
        rules = self.rules
        if word in self.dic:
            return word
        for rule in rules:
            if '*' in rule[0] and '?' in rule[0]:
                pattern = '(\w+)(\w)\\2' + rule[0][3:] + '$'
                repl = '\g<1>\g<2>' + rule[1][2:]
                a_word = re.sub(pattern,repl,word)
                if a_word in self.dic:
                    return a_word
            elif '*' in rule[0]:
                pattern = '(\w+)' + rule[0][1:] + '$'
                repl = '\g<1>' + rule[1][1:]
                a_word = re.sub(pattern,repl,word)
                if a_word in self.dic:
                    return a_word
        return word
    
def main():
    s = Stem()
    print s.stem('swimmed')

if __name__ == "__main__":
    main()
