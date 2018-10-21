import input_data
import tensorflow as tf


def getdataset():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    return mnist
def model():
    x = tf.placeholder("float", [None, 784])  # x是一个占位符 none表示第一个维度可以是任意值
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))# variable表示一个可以修改的张量
    y = tf.nn.softmax(tf.matmul(x, w) + b)  # 实现模型
    return x,w,b,y
def loss(y):
    y_ = tf.placeholder("float",[None,10])#标签
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))#loss
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    return train_step,y_
def run(x,y_,mnist,train_step):
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    return sess
def accuracy(sess,x,y,y_,mnist):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def main():
    mnist=getdataset()
    x,w,b,y=model()
    train_step,y_=loss(y)
    sess=run(x,y_,mnist,train_step)
    accuracy(sess,x,y,y_,mnist)

if __name__ == '__main__':
    main()
