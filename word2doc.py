from data_process import Word
from collections import Counter
import math
import codecs
from sklearn.decomposition import PCA
class Word2Doc:
    def __init__(self):
        self.data=Word().data
        self.mincount=5
        self.path='./word2doc.txt'
    def all_words(self):
        all_words=[]
        for line in self.data:
            all_words.extend(line)
        all_words=[item[0] for item in Counter(all_words).items() if item[1]>self.mincount]
        return all_words

    def idf(self):
        idf={}
        line_count=len(self.data)
        for line in self.data:
            for word in Counter(line).keys():
                if word in idf:
                    idf[word]+=1
                else:
                    idf[word]=1
        idf={word:math.log(line_count/(word_idf+1)) for word,word_idf in idf.items()}
        return idf

    def tf_idf(self,all_words,idf):
        tf_idf=[]
        for word in all_words:
            temp=[]
            for line in self.data:
                line_dict=dict(Counter(line))
                sum_word=sum(line_dict.values())
                word_tf=line_dict.get(word,0)/sum_word
                temp.append(word_tf*idf[word])
            tf_idf.append(temp)
        return tf_idf

    def dimension_reduction(self,tf_idf):
        pca=PCA(n_components=100)
        low_embedding=pca.fit_transform(tf_idf)
        return low_embedding

    def save_vector(self,all_words,low_embedding):
        output=codecs.open(self.path,'w','utf-8')
        for i in range(len(all_words)):
            output.write(all_words[i]+'\t'+','.join([str(vec) for vec in low_embedding[i]])+'\n')
        output.close()

    def get_vector(self):
        all_words=self.all_words()
        idf=self.idf()
        tf_idf=self.tf_idf(all_words,idf)
        low_embedding=self.dimension_reduction(tf_idf)
        self.save_vector(all_words,low_embedding)

word2doc=Word2Doc()
word2doc.get_vector()




