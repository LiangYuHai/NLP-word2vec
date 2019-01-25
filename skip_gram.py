import tensorflow as tf
from data_process import Word
from collections import Counter,deque
import math
import numpy as np
import codecs
class Skip_Gram:
    def __init__(self):
        self.batch_size=200
        self.embedding_dim=100
        self.num_sampled=100
        self.words=Word().data
        self.window_size=2
        self.LR=0.001
        self.mincount=5
        self.index=0
        self.output_path='./skip_gram.txt'

    def generate_word_dict(self):
        all_word=[]
        for line in self.words:
            all_word.extend(line)
        valid_word=[item[0] for item in Counter(all_word).items() if item[1]>self.mincount]
        word2id=dict()
        for i in range(1,len(valid_word)+1):
            word2id[valid_word[i-1]]=i
        word2id.update({'unknow':0})
        id2word=dict(zip(word2id.values(),word2id.keys()))

        train_data=[]
        for i in all_word:
            if i in word2id:
                train_data.append(word2id[i])
            else:
                train_data.append(0)
        return word2id,id2word,train_data

    def generate_batch(self,train_data):
        np_x=np.ndarray(shape=[self.batch_size],dtype=np.int32)
        np_y=np.ndarray(shape=[self.batch_size,1],dtype=np.int32)
        buffer=deque(maxlen=self.window_size*2+1)

        for i in range(self.window_size*2+1):
            buffer.append(train_data[self.index])
            self.index=(self.index+1)%len(train_data)

        for i in range(self.batch_size//(self.window_size*2)):
            for j in range(self.window_size*2+1):
                if j < self.window_size:
                    np_y[i*self.window_size*2+j][0] = buffer[j]
                    np_x[i*self.window_size*2+j] = buffer[self.window_size]
                elif j > self.window_size:
                    np_y[i * self.window_size * 2 + j-1][0] = buffer[j]
                    np_x[i * self.window_size * 2 + j-1] = buffer[self.window_size]
                else:
                    continue
            buffer.append(train_data[self.index])
            self.index = (self.index + 1) % len(train_data)

        return np_x,np_y


    def generate_word_vector(self,word2id,train_data):
        vocabulary_size=len(word2id)
        tf_x=tf.placeholder(tf.int32,[self.batch_size])
        tf_y=tf.placeholder(tf.int32,[self.batch_size,1])
        embedding=tf.Variable(tf.random_uniform(shape=[vocabulary_size,self.embedding_dim],minval=-1.0,maxval=1.0))
        word_embedding=tf.nn.embedding_lookup(embedding,tf_x)
        W=tf.get_variable('Weigth',shape=[vocabulary_size,self.embedding_dim],dtype=tf.float32,
                          initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(self.embedding_dim)))
        b=tf.get_variable('biases',[vocabulary_size],tf.float32,tf.zeros_initializer)
        loss=tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=W,
                                       biases=b,
                                       labels=tf_y,
                                       inputs=word_embedding,
                                       num_sampled=self.num_sampled,
                                       num_classes=vocabulary_size,
                                       )
        )
        train_op=tf.train.AdamOptimizer(self.LR).minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), axis=1, keepdims=True))
        normalize_embeddings = embedding / norm

        sess=tf.Session()
        init=tf.global_variables_initializer()
        sess.run(init)

        average_loss=0
        for i in range(1,5000):
            batch_x,batch_y=self.generate_batch(train_data)
            loss_,_=sess.run([loss,train_op],feed_dict={tf_x:batch_x,tf_y:batch_y})
            average_loss+=loss_
            if i%1000==0:
                average_loss/=1000
                print('iter:',i,"| average_loss:%.4f"%average_loss)
        word_vertors=sess.run(normalize_embeddings)
        return word_vertors

    def save_word_vectors(self,word_vectors,id2word):
        output=codecs.open(self.output_path,'w','utf-8')
        for index,vectors in enumerate(word_vectors):
            output.write(id2word[index]+'\t'+','.join([str(vec) for vec in vectors])+'\n')
        output.close()

    def train(self):
        word2id,id2word,train_data=self.generate_word_dict()
        word_vectors=self.generate_word_vector(word2id,train_data)
        self.save_word_vectors(word_vectors,id2word)
skip_gram=Skip_Gram()
skip_gram.train()
