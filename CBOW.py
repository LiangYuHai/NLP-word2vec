from data_process import Word
import tensorflow as tf
import numpy as np
class CBOW:
    def __init__(self,batch,mincount):
        self.words=Word()
        self.batch=batch
        self.mincount=mincount
        self.index=0

    def word_dict(self,words):
        all_words=[]
        for i in words:
            all_words.extend(i)
        from collections import Counter
        word_dict={item[0]:index for index,item in enumerate(Counter(all_words).items(),start=1) if item[1]>self.mincount}
        data=[]
        for line in words:
            temp=[]
            for word in line:
                if word in word_dict:
                    temp.append(word_dict[word])
                else:
                    temp.append(0)
            data.extend(temp)
        return word_dict,data


    def generate_batch(self,data):
        from collections import deque
        batch=200
        window_size=2
        span=window_size*2+1
        x=np.ndarray(shape=[batch,window_size*2])
        y=np.ndarray(shape=[batch,1])
        buffer=deque(maxlen=span)

        for _ in range(span):
            buffer.append(data[self.index])
            self.index=(self.index+1)%len(data)

        for i in range(batch):
            for j in range(window_size*2):
                if j!=window_size:
                    x[i][j]=buffer[j]
                else:
                    y[i][j]=buffer[j]
            buffer.append(data[self.index])
            self.index=(self.index+1)%len(data)

        assert x.shape[0]==batch and x.shape[1]==span-1

        return x,y

    def graph(self,word_dict):
        import math
        batch=200
        window_size=2
        vocabulary=len(word_dict)+1
        embedding_dim=100
        tf_x=tf.placeholder(tf.int32,[batch,window_size*2])
        tf_y=tf.placeholder(tf.int32,[batch,1])
        embedding=tf.random_uniform([vocabulary,embedding_dim],-1,1)
        W=tf.get_variable('weigth',shape=[vocabulary,embedding_dim],dtype=tf.float32,
                          initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(embedding_dim)))
        b=tf.get_variable('bias',shape=[vocabulary,1],dtype=tf.float32,initializer=tf.zeros_initializer)
        context_embedding=[]
        for i in range(window_size*2):
            context_embedding.append(tf.nn.embedding_lookup(embedding_dim,tf_x[:,i]))



