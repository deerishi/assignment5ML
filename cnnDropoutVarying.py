import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
    

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    

W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x_image=tf.reshape(x,[-1,28,28,1])

h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)


W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])


y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

dropouts=[0.25, 0.5, 0.75 ,1.0]
trainplt=[]
testplt=[]
for dropout in dropouts:
    print 'The dropout being used is ',dropout
    for i in range(1000):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: dropout})
    
    trainacc=[]
    batch_size=1000
    for i in range(mnist.train.num_examples/batch_size):
        batch = mnist.train.next_batch(1000)
  
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        trainacc.append(train_accuracy)  
        #print("step %d, training accuracy %g"%(i, train_accuracy))
    
    print 'mean training accuracy with dropout=',dropout ,' is  ',np.mean(trainacc)
    trainplt.append(np.mean(trainacc))
    
    testacc=[]
    batch_size=100
    for i in range(mnist.test.num_examples/batch_size):
        batch=mnist.test.next_batch(batch_size)
        acc=accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        #print("test accuracy %g"%acc)
        testacc.append(acc)
        
    print 'the mean test accuracy with dropout ',dropout,' is ',np.mean(testacc),'\n\n'
    testplt.append(np.mean(testacc))

plt.plot(dropouts,trainplt,label='Train set Accuracy')
plt.plot(dropouts,testplt,label='Test set Accuracy')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('keep_prob')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Dropout Variation ')

plt.show()
