import tensorflow as tf
import numpy as np

#Data
x_train = (np.arange(20).reshape((20,1))*np.ones((20,10))).reshape((20,10))
print(x_train[0].shape)

y_train = (np.arange(20)*0.5).reshape((20,1))
print(y_train.shape)


# Computation Graph
x = tf.placeholder(tf.float32,shape=(None,10),name='x')
y = tf.placeholder(tf.float32,shape=(None,1),name='y')
w = tf.Variable(np.ones((10,1),dtype=np.float32),name='w')
b = tf.Variable(np.ones((1),dtype=np.float32),name='b')

y_pred = tf.add(tf.matmul(x,w),b,name='y_pred')
cost = tf.reduce_sum(tf.pow(y-y_pred,2))/20

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

# Running the grapg in session with appropriate parameters
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("trial1")
    writer.add_graph(sess.graph)

    for epoch in range(1000):
        for (X,Y) in zip(x_train,y_train):
            sess.run(optimizer,feed_dict={x:X.reshape((1,10)),y:Y.reshape(1,1)})
        if((epoch+1)%100 == 0):
            c = sess.run(cost,feed_dict={x:x_train,y:y_train})
            print("cost:",c,"weight:",sess.run(w),"bias:",sess.run(b))

    print("training done...\n")
    training_cost = sess.run(cost,feed_dict={x:x_train,y:y_train})
    print("training_cost:"+str(training_cost)+'\n')
