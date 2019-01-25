from data_process import Word
import collections
import codecs
from sklearn.decomposition import PCA
class Word2Word:
    def __init__(self):
        self.data=Word().data
        self.window_size=5
        self.mincount=5
        self.path='./word2word.txt'

    def all_words(self):
        all_words=[]
        for line in self.data:
            all_words.extend(line)
        all_words=[item[0] for item in collections.Counter(all_words).items() if item[1]>self.mincount]
        return all_words

    def generate_word_dict(self):
        word2word_dict={}
        for line in self.data:
            for index,word in enumerate(line):
                if index<self.window_size:
                    left=line[:index]
                else:
                    left=line[index-self.window_size:index]
                if index+self.window_size>len(line):
                    right=line[index+1:]
                else:
                    right=line[index+1:index+self.window_size+1]
                context=left+right
                if word not in word2word_dict:
                    word2word_dict[word]={}
                else:
                    for context_word in context:
                        if context_word not in word2word_dict[word]:
                            word2word_dict[word][context_word]=1
                        else:
                            word2word_dict[word][context_word]+=1
        return word2word_dict

    def generate_vector_matrix(self,all_words,word2word_dict):
        vector_matrix=[]
        for center_word in all_words:
            weigth_sum=sum(word2word_dict[center_word].values())
            temp=[]
            for context_word in all_words:
                weigth=word2word_dict[center_word].get(context_word,0)/weigth_sum
                temp.append(weigth)
            vector_matrix.append(temp)
        return vector_matrix

    def dimensionality_reduction(self,vector_matrix):
        pca=PCA(n_components=100)
        low_embedding=pca.fit_transform(vector_matrix)
        return low_embedding

    def save_word_vector(self,all_words,vector_matrix):
        output=codecs.open(self.path,'w','utf-8')
        for i in range(len(all_words)):
            output.write(all_words[i]+'\t'+','.join([str(vec) for vec in vector_matrix[i]])+'\n')
        output.close()

    def get_vector(self):
        all_words=self.all_words()
        word2word_dict=self.generate_word_dict()
        vector_matrix=self.generate_vector_matrix(all_words,word2word_dict)
        low_embedding=self.dimensionality_reduction(vector_matrix)
        self.save_word_vector(all_words,low_embedding)

word2word=Word2Word()
word2word.get_vector()


