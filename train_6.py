#coding=utf-8
#import xlrd
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
#import time
#import datetime
tf.reset_default_graph() 
rnn_unit=10       #隐层数量
input_size=6
output_size=1
lr=0.0006         #学习率
#——————————————————导入训练数据——————————————————————
data0 = pd.read_csv('Data23.csv',index_col=None,parse_dates = True) #pd.read_csv默认生成DataFrame对象
data2 = data0.iloc[:-227,-1].values[:,np.newaxis]
data3 = data0.iloc[227:,2:8].values  #取第2-9列
data = np.concatenate((data2,data3),axis=1)
data4 = data[:,-1]
#获取训练集
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=70000):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:6]
       y=normalized_train_data[i:i+time_step,6,np.newaxis]
       where_are_nan = np.isnan(x) 
       x[where_are_nan] = 0 
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#——————————————————定义神经网络——————————————————
def lstm(X):
    
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input_hi=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input_hi,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,states

#————————————————训练模型————————————————————

#def train_lstm(batch_size=60,time_step=20,train_begin=0,train_end=4000):
batch_size=60
time_step=20
train_begin=0
train_end=70000
X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
with tf.variable_scope("sec_lstm"):
    pred,_=lstm(X)
loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
train_op=tf.train.AdamOptimizer(lr).minimize(loss)
saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
        for step in range(len(batch_index)-1):
            _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
        print("Number of iterations:",i," loss:",loss_)
    print("model_save10: ",saver.save(sess,'model_save10\\modle.ckpt'))
    print("v1 : %s" % weights['in'].eval())
    print("v2 : %s" % biases['in'].eval())
    print("v1 : %s" % weights['out'].eval())
    print("v2 : %s" % biases['out'].eval())
    #我是在window下跑的，这个地址是存放模型的地方，模型参数文件名为modle.ckpt
    #在Linux下面用 'model_save2/modle.ckpt'
    print("The train has finished")
#train_lstm()
