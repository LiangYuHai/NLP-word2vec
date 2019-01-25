import numpy as np
import codecs
import sys
class Show:
    def __init__(self,path):
        self.path=path
        self.num_show=10
        self.word2vector,self.all_vectors,self.id2word = self.read(self.path)

    def read(self,path):
        input=codecs.open(path,'r','utf-8')
        word2vector=dict()
        all_vectors=[]
        id2word={}
        index=0
        for line in input.readlines():
            line=line.strip().split('\t')
            vector=np.asarray([float(vec) for vec in line[1].split(',') if vec])
            all_vectors.append(vector)
            word2vector[line[0]]=vector
            id2word[index]=line[0]
            index+=1
        return word2vector,np.asarray(all_vectors),id2word

    def showtime(self,word):
        if word in self.word2vector:
            vector=self.word2vector[word]
            all_vector=self.all_vectors.T
            dot_product=np.dot(vector,all_vector)
            norm=np.linalg.norm(vector,ord=2)*np.linalg.norm(all_vector,ord=2)
            cos=dot_product/norm
            similars=0.5+0.5*cos
            similar_dict={self.id2word[index]:similar for index, similar in enumerate(similars.tolist()) if self.id2word[index]!=word}
            similar_words=sorted(similar_dict.items(),key=lambda x:x[1],reverse=True)[:self.num_show]
            return similar_words
        else:
            print('The word not in dict')



if len(sys.argv)==2 and sys.argv[1]=='cbow':
    path='./cbow.txt'
    show=Show(path)
    word = input('输入一个词：')
    similar_words=show.showtime(word)
    words=[]
    for item in similar_words:
        words.append(item[0])
    print(word+':'+','.join(words))

if len(sys.argv)==2 and sys.argv[1]=='skip_gram':
    path='./skip_gram.txt'
    show=Show(path)
    word=input('输入一个词：')
    similar_words = show.showtime(word)
    words = []
    for item in similar_words:
        words.append(item[0])
    print(word + ':' + ','.join(words))
