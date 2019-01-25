from data_process import Word
import tensorflow as tf
import numpy as np
import codecs
class CBOW:
    def __init__(self):
        self.words=Word().data
        self.batch=200
        self.mincount=5
        self.word_index=0
        self.ITER=5000
        self.window_size=2
        self.num_sampled=100
        self.LR=0.001
        self.embedding_dim=100
        self.output_path='./cbow.txt'

    def generate_word_dict(self):
        all_words=[]
        for i in self.words:
            all_words.extend(i)
        from collections import Counter
        word_dict=[item[0] for item in Counter(all_words).items() if item[1]>self.mincount]
        word2id={}
        for i in range(1,len(word_dict)+1):
            word2id[word_dict[i-1]]=i
        word2id.update({'unknow':0})
        id2word=dict(zip(word2id.values(),word2id.keys()))
        train_data=[]
        for line in self.words:
            temp=[]
            for word in line:
                if word in word2id:
                    temp.append(word2id[word])
                else:
                    temp.append(0)
            train_data.extend(temp)
        return word2id,id2word,train_data

    def generate_batch(self,train_data):
        from collections import deque
        span=self.window_size*2+1
        x=np.ndarray(shape=[self.batch,self.window_size*2])
        y=np.ndarray(shape=[self.batch,1])
        buffer=deque(maxlen=span)

        for _ in range(span):
            buffer.append(train_data[self.word_index])
            self.word_index=(self.word_index+1)%len(train_data)

        for i in range(self.batch):
            col_index=0
            for j in range(span):
                if j!=self.window_size:
                    x[i][col_index]=buffer[j]
                    col_index+=1
                else:
                    y[i][0]=buffer[j]
            buffer.append(train_data[self.word_index])
            self.word_index=(self.word_index+1)%len(train_data)

        return x, y

    def build_net(self, word2id, train_data):
        import math
        vocabulary_size=len(word2id)
        tf_x=tf.placeholder(tf.int32, [self.batch, self.window_size*2])
        tf_y=tf.placeholder(tf.int32, [self.batch, 1])
        embedding=tf.Variable(tf.random_uniform([vocabulary_size, self.embedding_dim], -1.0, 1.0))
        W=tf.get_variable('weigth', shape=[vocabulary_size, self.embedding_dim], dtype=tf.float32,
                          initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(self.embedding_dim)))
        b=tf.get_variable('bias', shape=[vocabulary_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        context_embedding=[]
        for i in range(self.window_size*2):
            context_embedding.append(tf.nn.embedding_lookup(embedding, tf_x[:, i]))
        context_embedding=tf.reduce_mean(tf.stack(context_embedding, axis=0), axis=0, keepdims=False)
        loss=tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=W,
                                        biases=b,
                                        inputs=context_embedding,
                                        labels=tf_y,
                                        num_sampled=self.num_sampled,
                                        num_classes=vocabulary_size,
                                        )
        )

        train_op=tf.train.AdamOptimizer(self.LR).minimize(loss)
        norm=tf.sqrt(tf.reduce_sum(tf.square(embedding),axis=1,keepdims=True))
        normalize_embeddings=embedding/norm

        sess=tf.Session()
        init=tf.global_variables_initializer()
        sess.run(init)

        average_loss = 0
        for iter in range(1,self.ITER+1):
            batch_x,batch_y=self.generate_batch(train_data)
            loss_,_=sess.run([loss,train_op],feed_dict={tf_x:batch_x,tf_y:batch_y})
            average_loss+=loss_
            if iter%1000==0:
                average_loss=average_loss/1000
                print('iter:',iter,'| average_loss:%.4f'%average_loss)
                average_loss=0
        words_vectors=sess.run(normalize_embeddings)
        return words_vectors

    def save_vec(self,words_vectors,id2word):
        output=codecs.open(self.output_path,'w','utf-8')
        for index,vector in enumerate(words_vectors):
            output.write(id2word[index]+'\t'+','.join([item for item in vector])+'\n')
        output.close()

    def train(self):
        word2id,id2word,train_data=self.generate_word_dict()
        words_vectors=self.build_net(word2id,train_data)
        self.save_vec(words_vectors,id2word)

cbow=CBOW()
vec=cbow.train()

