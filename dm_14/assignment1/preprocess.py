#coding:utf-8

import jieba
import os
import sys
import re

reload(sys)
sys.setdefaultencoding('utf-8')

with open('stopwords.txt') as f:
    stopwords = [line.strip().decode('utf-8') for line in f.xreadlines()]
    stopwords = set(stopwords)

def filter_stopwords(words):
    return filter(lambda x: x not in stopwords,words)

def main():
    lily_objs = list()
    lily_files = ['lily/'+file for file in os.listdir('lily')]
    for lily_file in lily_files:
        with open(lily_file) as f:
            label = lily_file.split('/')[1][:-4]
            for line in f.xreadlines():
                lily_obj = dict()
                content = line.strip().decode('utf-8')
                content = content.replace('\t','\s')
                content = re.sub('\s{2,}','\s',content)
                words = list(jieba.cut(content))
                words = filter_stopwords(words)
                words = [word.lower() for word in words]
                lily_obj['label'] = label
                lily_obj['content'] = content
                lily_obj['words'] = words
                lily_objs.append(lily_obj)

    with open('lily.txt','w') as f:
        for lily_obj in lily_objs:
            label = lily_obj['label']
            words = lily_obj['words']
            line = '%s\t%s\n' % (label,' '.join(words))
            line = line.encode('utf-8')
            f.write(line)
        
if __name__ == "__main__":
    main()
