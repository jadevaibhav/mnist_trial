import tensorflow as tf
import numpy as np

#Data
x_train = np.arange(20)
print(x_train)

y_train = (np.arange(20)*0.5)
print(y_train)

#linear regression in numpy
w1 = 1
b1 = 1
def forward_pass(x,y,w,b):
     y_pred = x*w + b
     cost = (y-y_pred)**2
     cost = cost/20
     return cost,y_pred

def backward_pass(cost,y_pred,x,y,w,b):
    del_y_pred = cost/(2*y_pred - 2*y+0.99)
    del_w = del_y_pred/(x+0.99)
    del_b = del_y_pred/0.99
    return del_w,del_b

n_epochs = 1000
lr = 0.01
cost1 = y_pred1 = 0
for i in range(n_epochs):
    update_w1 = update_b1 = 0
    prev_del_w1 = prev_del_b1 = 1000
    for x1,y1 in zip(x_train,y_train):
        cost1,y_pred1 = forward_pass(x1,y1,w1,b1)
        del_w1,del_b1 = backward_pass(cost1,y_pred1,x1,y1,w1,b1)
        #if(prev_del_w1>del_w1+50 and prev_del_b1>=del_b1+50):
        update_w1 = update_w1 + del_w1
        update_b1 = update_b1 + del_b1
        prev_del_w1 = del_w1
        prev_del_b1 = del_b1
    w1 = w1 - lr*np.sign(w1)*update_w1
    b1 = b1 - lr*np.sign(b1)*update_b1
    if(i%100 == 0):
        f_cost = 0
        for x1,y1 in zip(x_train,y_train):
            cost1,y_pred1 = forward_pass(x1,y1,w1,b1)
            f_cost = f_cost + cost1
        print("cost:",f_cost,"weight:",w1,"bias:",b1)
print("training_cost:",f_cost)

# Computation Graph
x = tf.placeholder(tf.float32,name='x')
y = tf.placeholder(tf.float32,name='y')
w = tf.Variable(np.ones((1),dtype=np.float32),name='w')
b = tf.Variable(np.ones((1),dtype=np.float32),name='b')

y_pred = tf.add(tf.multiply(x,w),b,name='y_pred')
cost = tf.reduce_sum(tf.pow(y-y_pred,2))/20

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

# Running the grapg in session with appropriate parameters
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("trial")
    writer.add_graph(sess.graph)

    for epoch in range(1000):
        for (X,Y) in zip(x_train,y_train):
            sess.run(optimizer,feed_dict={x:X,y:Y})
        if((epoch+1)%100 == 0):
            c = sess.run(cost,feed_dict={x:x_train,y:y_train})
            print(c,sess.run(w),sess.run(b))

    print("training done...\n")
    training_cost = sess.run(cost,feed_dict={x:x_train,y:y_train})
    print("training_cost:"+str(training_cost)+'\n')
