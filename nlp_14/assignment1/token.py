#coding:utf-8

class Tokenlizer(object):
    def __init__(self):
        self.max_len = 5
        self.dic = set()
        with open('data/dic.txt') as f:
            for line in f:
                line = line.strip().decode('utf-8')
                word = line.split(',')[0]
                self.dic.add(word)

    def fmm(self,sentence):
        words = list()
        s_len = len(sentence)
        while s_len>0:
            max_len = self.max_len if s_len>self.max_len else s_len
            for i in range(max_len,0,-1):
                word = sentence[:i]
                if word in self.dic or i==1:
                    words.append(word)
                    sentence = sentence[i:]
                    s_len = len(sentence)
                    break
        return words
    
    def rmm(self,sentence):
        words = list()
        s_len = len(sentence)
        while s_len>0:
            max_len = self.max_len if s_len>self.max_len else s_len
            for i in range(max_len,0,-1):
                word = sentence[s_len-i:]
                if word in self.dic or i==1:
                    words.append(word)
                    sentence = sentence[:s_len-i]
                    s_len = len(sentence)
        words.reverse()
        return words

def main():
    tokenlizer = Tokenlizer()
    sentence = u'我想去阿富汗吃晚饭'
    words = tokenlizer.fmm(sentence)
    print ' '.join(words)
    words = tokenlizer.rmm(sentence)
    print ' '.join(words)

if __name__ == "__main__":
    main()
