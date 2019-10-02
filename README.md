# 建立简单的CNN模型并使用MNIST数据集训练模型并测试

### 在tensorflow下利用mnist数据集来训练一个卷积神经网络识别手写数字

##### 首先我们导入一些必要的包：

```python
import input_data
import tensorflow as tf
```

input_data 是一个读数据的包在里面写好了读数据需要用到的一些基本函数

##### 然后我们来读取数据到主函数：

```python
def getdata():
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)#读数据
    return mnist
```

##### 然后建立卷积神经网络需要用到的一些函数：

```python
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):#设置卷积层 x为输入张量，w为卷积核，步长为1，考虑边界不足部0
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# padding 设置成了same 就使图片在卷积后不改变尺寸

def max_pool_2x2(x):#池化 x是需要池化的输入 ksize窗口大小长为2宽为2 strides为步长设为2*2 考虑边界
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],\
                          strides=[1, 2, 2, 1], padding='SAME')
```

weighr_variable()和bias_variable()函数分别初始化权重和偏置量 

conv2d()函数用来设置卷积层  x为输入张量，w为卷积核，步长为1，考虑边界不足部0

max_pool_2x2(x)用来设置最大池化 x是需要池化的输入 ksize窗口大小长为2宽为2 strides为步长设为2*2 考虑边界

##### 下面来建立卷积神经网络的模型：

```python
def cnn_model(x):
    #第一层卷积 卷积在每个5*5的patch中算出32个特征
    w_conv1 = weight_variable([5, 5, 1, 32])#卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量。
    b_conv1 = bias_variable([32])#偏置量
    x_image = tf.reshape(x,[-1,28,28,1])#把图片转换成4维向量
    h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)#把输入值进行卷积加上偏置项再用relu激活
    h_pool1 = max_pool_2x2(h_conv1)#再池化
    #第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])#把几个类似的层堆叠起来得到64给特征
    b_conv2 = bias_variable([64])#设置64个偏置项
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#卷积加偏置项再激活
    h_pool2 = max_pool_2x2(h_conv2)#进行池化
    #设置密集连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])#经过第一次卷积图片尺寸不变  再池化变为14*14 再卷积尺寸不变 再池化变为7*7 然后有64个特征
    b_fc1 = bias_variable([1024])#由于我们加入了一个1024个神经元的全连接层，就设置了1024个偏置项
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])#转换数据类型
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)#乘上权重再加上偏置项再激活
    keep_prob = tf.placeholder("float")#为了防止过拟合 我们在输出层之前加入dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #输出层
    return y_conv,keep_prob
```



##### 然后来进行模型训练：

```python
def runcnn(keep_prob,sess,x,y_conv,y_,mnist):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))    #交叉熵
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)   #训练
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  #判断正误
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  #输出准确度
    sess.run(tf.initialize_all_variables()) #初始化变量
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:   #每一百次输出结果
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print("running with cnn accuracy %g" % accuracy.eval(feed_dict={\
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

在训练中我们用交叉熵来定义损失函数，利用adam优化方法，设置学习速率为1e-4，最小化交叉熵。

每循环100次输出准确率。

#####  最后我们来写主程序：

```python
def main():
    mnist = getdata()
    sess, x, w, y_, b, y=model()
    cross_entropy=loss(y,y_)
    run(x,cross_entropy,y_,mnist)
    accuracy(y,y_,x,mnist)
    y_conv,keep_prob=cnn_model(x)
    runcnn(keep_prob,sess,x,y_conv,y_,mnist)
```

最后测试得准确率为0.9931 效果还不错。

## 总结

卷积神经网络用一个卷积核来提取特征再经过池化层可以非常有效地缩小参数矩阵的尺寸，从而减少最后全连层中的参数数量。使用池化层即可以加快计算速度也有防止过拟合的作用。一层神经网络由一个卷积层和一个池化层组成，有时候为了提高模型的复杂度我们可以构建多层神经网络。





