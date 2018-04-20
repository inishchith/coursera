import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)        # one set others unset 

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100    # number of features fed at one time 

# height X width = matrix 

x = tf.placeholder(tf.float32,[None,784])        # X * 784 (28*28 flatened) 
y = tf.placeholder(tf.float32)

def neural_network_model(data):
    
    # input_data * weights + biase (why? - > if input_data==0 , no neuron would fire ( just a linear eqn. intercept shift) 
    
    # updation - new_weight = old_wieight + learning_rate(desired_output-actual_output)*input

        
    hidden_layer1 = {'weights': tf.Variable(tf.random_normal([784,n_nodes_hl1])),   
                     'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_layer2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_layer3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_layer4 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl3])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                     'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    layer_1 = tf.add(tf.matmul(data,hidden_layer1['weights']),hidden_layer1['biases'])
    l1 = tf.nn.relu(layer_1)        # rectified linear i.e Threshold / activation function < just as SGD >

    layer_2 = tf.add(tf.matmul(l1,hidden_layer2['weights']),hidden_layer2['biases'])
    l2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(l2,hidden_layer3['weights']),hidden_layer3['biases'])
    l3 = tf.nn.relu(layer_3)

    layer_4 = tf.add(tf.matmul(l3,hidden_layer4['weights']),hidden_layer4['biases'])
    l4 = tf.nn.relu(layer_4)
    
    output = tf.matmul(l4,output_layer['weights']) + output_layer['biases']
    
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)    # one hot array
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    # softmax[i] = (e^x[i])/sum(e^x[])
    # cross_entropy = -sum(label[i]*logits[i])

    optimizer = tf.train.AdamOptimizer().minimize(cost)         # adam_optimizer default params - learning rate 0.001 (DEFAULT)

    epoch_count = 10       # number of cycles

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(epoch_count):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)

                pred,tp,c = sess.run([prediction,optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})

                epoch_loss += c
            #print(pred)
            #print(epoch_y)

            print("Epoch : ",(epoch+1)," completed of ",epoch_count)
            print("Epoch loss: ", epoch_loss)
        writer = tf.summary.FileWriter("./graph",sess.graph) 
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        
        print("Accuracy :",accuracy.eval({x:mnist.test.images,y:mnist.test.labels})*100)

train_neural_network(x)


'''
Approach : (feed forward neu-net) 

1. input > weights(along with) > hidden layer1 (activation function ) > weights(along with) > hidden layer2 (AF) > weights(along with) > output 

2. compare gen. output to intended output  > cost / loss function (cross entropy) 

3. use optimization function (optimizer) to minimize cost (AdamOptimizer , SGD , AdaGrad )

4. Go Back and manipulate weights ( BACK PROPAGATION) 

5. Feed Forward + back_prop = 1 epoch 



'''


