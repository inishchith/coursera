import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

c = a*b

with tf.Session() as sess:
    print(sess.run(c,feed_dict={a:5,b:10}))
    writer = tf.summary.FileWriter('./graph',sess.graph)
    writer.close()
    sess.close()


'''
$ tensorboard --logdir=graph/
'''
