import tensorflow as tf

node1 = tf.constant(43.0,dtype=tf.float32)
node2 = tf.constant(36.0) # implict
# print(node1,node2)
sess = tf.Session()
print(sess.run([node1,node2]))


node3 = tf.add(node1,node2)
#print(node3)
print(sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder = a+b

print(sess.run(adder,feed_dict={a:3,b:10}))
print(sess.run(adder,feed_dict={a:[1,4],b:[5,8]}))

# trainable model 

W = tf.Variable([5,1], dtype=tf.float32)
b = tf.Variable([-3,3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

initially = tf.global_variables_initializer()

print()
print("Initializers")
sess.run(initially)     # process initializer with params value
print(sess.run(linear_model,{x:[1,2]}))
print(sess.run(linear_model,feed_dict={x:[1,2],W:[3,4],b:[5,6]}))
print(sess.run(linear_model,{x:[0,0]}))

print()
print("Calculations")
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)        # W = [5,1] default  
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2], y: [-2, -3]}))      # linear_model = [2,5] - [-2,-3]
print(sess.run(squared_deltas,{x:[1,2],y:[-2,-3]}))


print()
print("tf.Variables can be reassigned using tf.assign")

norm = tf.random_normal([2,2])

print(sess.run([tf.reduce_sum(norm),norm]))
l1 = tf.nn.relu(x)

print(sess.run(l1,feed_dict={x:[-.1,-.2,-.3]}))
