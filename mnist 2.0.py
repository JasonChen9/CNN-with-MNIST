import input_data
import tensorflow as tf

def getdata():
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)#读数据
    return mnist

def model():
    sess = tf.InteractiveSession()#在创建图之前构造会话
    x  = tf.placeholder("float",shape=[None,784])#创建x占位符
    y_ = tf.placeholder("float",shape=[None,10])#创建y_占位符
    w  = tf.Variable(tf.zeros([784,10]))#初始化w
    b  = tf.Variable(tf.zeros([10]))#初始化b
    sess.run(tf.initialize_all_variables())#初始化变量
    y = tf.nn.softmax(tf.matmul(x, w) + b)#模型
    return sess,x,w,y_,b,y

def loss(y,y_):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))#定义交叉熵
    return cross_entropy

def run(x,cross_entropy,y_,mnist):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#学习率为0.01，最小化交叉熵
    for i in range(1000):
        batch = mnist.train.next_batch(50)#从数据集选50个数据
        train_step.run(feed_dict={x:batch[0],y_:batch[1]})#随机梯度下降

def accuracy(y,y_,x,mnist):
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))#argmax取一列的最大的一个值，再判断是否相等
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))#把返回的列表转化成float类型，并求平均值作为准确率
    print('running with tranditional sortmax,accuracy:%g'% accuracy.eval(feed_dict={x: mnist.test.images,y_:mnist.test.labels}))
#初始化权重和偏置项
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):#设置卷积层 x为输入张量，w为卷积核，步长为1，考虑边界不足部0
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):#池化 x是需要池化的输入 ksize窗口大小长为2宽为2 strides为步长设为2*2 考虑边界
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],\
                          strides=[1, 2, 2, 1], padding='SAME')

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
    W_fc1 = weight_variable([7 * 7 * 64, 1024])#经过第一次卷积图片变为24*24 再池化变为12*12 再卷积变为8*8 再池化变为7*7 然后有64个特征
    b_fc1 = bias_variable([1024])#由于我们加入了一个1024个神经元的全连接层，就设置了1024个偏置项
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])#转换数据类型
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)#乘上权重再加上偏置项再激活
    keep_prob = tf.placeholder("float")#为了防止过拟合 我们在输出层之前加入dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #输出层
    return y_conv,keep_prob
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

def main():
    mnist = getdata()
    sess, x, w, y_, b, y=model()
    cross_entropy=loss(y,y_)
    run(x,cross_entropy,y_,mnist)
    accuracy(y,y_,x,mnist)
    y_conv,keep_prob=cnn_model(x)
    runcnn(keep_prob,sess,x,y_conv,y_,mnist)



if __name__=='__main__':
    main()
