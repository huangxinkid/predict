#coding=utf-8
#import xlrd
#import sys
from scipy import stats
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import tensorflow as tf
tf.reset_default_graph() 
rnn_unit=10       #隐层数量
input_size=6
output_size=1
lr=0.0006         #学习率
#——————————————————导入测试数据——————————————————————
#data01 = sys.argv[1]
#data00 = pd.read_csv(data01,index_col=None,parse_dates = True) #pd.read_csv默认生成DataFrame对象

data00 = pd.read_csv('Data_pre.csv',index_col=None,parse_dates = True) #pd.read_csv默认生成DataFrame对象

date = data00.iloc[0,1]
t_date=datetime.strptime(date, '%Y/%m/%d %H:%M')
n_days =t_date + timedelta(days=1)
date = n_days.strftime('%Y-%m-%d %H:%M:%S')

data20 = data00.iloc[:-227,-1].values[:,np.newaxis]
data30 = data00.iloc[227:,2:8].values  #取第2-9列
data01 = np.concatenate((data20,data30),axis=1)

#获取测试集
def get_test_data(time_step=20,test_begin=0,test_end=499):
    data_test=data01[test_begin:test_end]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    normalized_test_data[np.isnan(normalized_test_data)] = 0 #将标准化后数值中的NAN转为0
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:6]
       y=normalized_test_data[i*time_step:(i+1)*time_step,6]
#       x[np.isnan(x)] = 0 
       test_x.append(x.tolist())
       test_y.extend(y)
       
    test_x.append((normalized_test_data[(i+1)*time_step:,:6]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,6]).tolist())
    return mean,std,test_x,test_y
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
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, reuse=None, forget_bias=1.0, state_is_tuple=True)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#————————————————预测模型————————————————————
#def prediction(time_step=20):
#data002 = sys.argv[2]   
#data003 = sys.argv[3]
    
time_step=20
X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
mean,std,test_x,test_y=get_test_data(time_step)
with tf.variable_scope("sec_lstm"):
    pred,_=lstm(X)
saver=tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    #参数恢复
    module_file = tf.train.latest_checkpoint('model_save10')
    saver.restore(sess, module_file)
    test_predict=[]
    for step in range(len(test_x)-1):
        prob=sess.run(pred,feed_dict={X:[test_x[step]]})
        predict=prob.reshape((-1))
        test_predict.extend(predict)
    test_y=np.array(test_y)*std[6]+mean[6]
    test_predict=np.array(test_predict)*std[6]+mean[6]
    acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差程度
    print("The accuracy of this predict:",acc)
    
    x1=np.array(test_predict)
    if len(test_y) % time_step != 0:        
        x1= np.concatenate((x1,test_y[-(len(test_y) % time_step):]),axis=0)
    else:
        x1= np.concatenate((x1,test_y[-time_step:]),axis=0)
    mean=x1.mean()        
    std=x1.std()
    a_list=[]
    b_list=[]
    for i in x1:        
        interval=stats.t.interval(0.80,len(x1)-1,i,std)
        a=interval[0]
        b=interval[1]
        a_list.append(a)
        b_list.append(b)
    plt.figure(figsize=(15, 5))
#    plt.plot(list(range(len(a_list))), a_list,  color='b')
#    plt.plot(list(range(len(b_list))), b_list,  color='b') 
    plt.plot(list(range(len(x1))), x1, color='g')
    plt.plot(list(range(len(test_y))), test_y,  color='r')
#    plt.axvline(70000, linestyle="dotted", linewidth=4, color='g')
#    plt.plot(range(69500,69500+len(data4[69500:70000])),data4[69500:70000],color='r')
    date_pre = pd.date_range(date,periods=len(a_list),freq='5T')    
    dataframe = pd.DataFrame({'date':date_pre,'up':a_list,'down':b_list})    #列表转化成dataframe
    dataframe.to_csv('dataQ8.csv',index=False)

