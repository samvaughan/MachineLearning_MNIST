import tensorflow as tf
import datetime

#Constants we'll need
#We'll have a 748, 300, 100, 10 network
n_inputs=28*28
n_hidden1=300
n_hidden2=100
n_outputs=10


learning_rate=0.01
dropout_rate=0.5
n_epochs=75
batch_size=50

#Reset the graph at the beginning
tf.reset_default_graph()

#Training or not?
training=tf.placeholder_with_default(False, shape=(), name="training_bool")

#Plaeholders for the data
X=tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y=tf.placeholder(tf.int64, shape=(None), name='y')

#Logging set up 
now=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
root_logdir="tf_logs"
logdir="{}/run-{}/".format(root_logdir, now)


#The actual network 
with tf.name_scope('dnn'):

    #Layer 1 and dropout
    hidden1=tf.layers.dense(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
    hidden1_drop=tf.layers.dropout(hidden1, dropout_rate, training=training)

    #Layer 2 and dropout
    hidden2=tf.layers.dense(hidden1_drop, n_hidden2, name='hidden2', activation=tf.nn.relu)
    hidden2_drop=tf.layers.dropout(hidden2, dropout_rate, training=training)

    #Output before softmax
    logits=tf.layers.dense(hidden2_drop, n_outputs, name='outputs')

#The loss function
#Use Cross Entropy to see how well we're doing
with tf.name_scope('loss'):
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss=tf.reduce_mean(cross_entropy, name='loss')
    cross_ent_summary=tf.summary.scalar('cross_entropy', cross_entropy)

#Adam optimiser with default hyperparameters
with tf.name_scope('train'):
    optimiser=tf.train.AdamOptimizer()
    training_op=optimiser.minimize(loss)


#Evaluation
with tf.name_scope('eval'):
    correct=tf.nn.in_top_k(logits, y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct, tf.float32))
    acc_summary=tf.summary.scalar('Accuracy', accuracy)

#Summary
file_writer=tf.summary.FileWriter(logdir, tf.get_default_graph())


#Init and save
init = tf.global_variables_initializer()
saver=tf.train.Saver()



#Read in the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/")




with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples//batch_size):
            X_batch, y_batch=mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training:True})

            #Logging
            if iteration % 10 ==0:
                acc_summary_str=acc_summary.eval(feed_dict={X: X_batch, y: y_batch, training:True})
                step=epoch*(mnist.train.num_examples//batch_size)*iteration
                file_writer.add_summary(acc_summary_str, step)

        #Accuracy on the train and test data
        acc_train=accuracy.eval(feed_dict={X: X_batch, y: y_batch, training:True})
        acc_test=accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels, training:False})

        print "Epoch: {}. Train Accuracy: {}, Test Accuracy: {}".format(epoch, acc_train, acc_test)

    save_path = saver.save(sess, "./my_model.ckpt")


file_writer.close()