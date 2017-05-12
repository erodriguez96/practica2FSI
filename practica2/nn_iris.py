import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first columns of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last column. Then we encode them in one hot code

x_data_training = x_data[:106]
y_data_training = y_data[:106]

x_data_validating = x_data[106:128]
y_data_validating = y_data[106:128]

x_data_testing = x_data[128:150]
y_data_testing = y_data[128:150]



#print "\nSome samples..."
#for i in range(20):
#    print x_data[i], " -> ", y_data[i]
#print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

# 4 entradas y 5 neuronas
W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

#W1h = tf.Variable(np.float32(np.random.rand(5, 5)) * 0.1)
#b1h = tf.Variable(np.float32(np.random.rand(5) * 0.1))


W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)


h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20

for epoch in xrange(100):
    for jj in xrange(len(x_data_training) / batch_size):
        batch_xs = x_data_training[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_training[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    result = sess.run(y, feed_dict={x: x_data_validating})
    for b, r in zip(y_data_validating, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"

print "test"
print "Epoch #: Test", "Error: ", sess.run(loss, feed_dict={x: x_data_testing, y_:y_data_testing})
result = sess.run(y,feed_dict={x: x_data_testing})
for b, r in zip(y_data_testing, result):
    print b, "-->", r