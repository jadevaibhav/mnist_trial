import tensorflow as tf
import os
from PIL import Image
import numpy as np
path = 'trainingSample'
X = []
Y = []
for i in range(10):
    imgs = os.listdir(path+'/'+str(i))
    for j in imgs:
        img = Image.open(path+'/'+str(i)+'/'+j)
        img = np.asarray(img).reshape((28,28,1))
        X.append(img)
        y1 = np.zeros((10))
        y1[i] = 1
        Y.append(y1)
X = np.asarray(X)
Y = np.asarray(Y)

p = np.random.permutation(600)
X = X[p]
Y = Y[p]
x_train = X[:550]

y_train = Y[:550]
x_val = X[550:600]
y_val = Y[550:600]

n_classes = 10
batch_size = 64

x = tf.placeholder('float', [None, 28,28,1])
y = tf.placeholder('float',[None,10])

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    max1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    max2 = maxpool2d(conv2)

    fc = tf.reshape(max2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']
    #print(output.shape)
    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    print('check here..'+str(prediction.shape)+'')
    cost = tf.reduce_mean( tf.losses.softmax_cross_entropy(onehot_labels=y,logits=prediction) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(x_train.shape[0]):
                epoch_x = x_train[_].reshape((1,28,28,1))
                epoch_y = y_train[_].reshape((1,10))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #print('Accuracy:',accuracy.eval({x:x_val, y:y_val}))

train_neural_network(x)
