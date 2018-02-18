import numpy as np 
import tensorflow as tf
import datetime


n_inputs=28*28
n_hidden1=300
n_hidden2=100
n_outputs=10
learning_rate=0.01
dropout_rate=0.5
n_epochs=75
batch_size=50


tf.reset_default_graph()
training=tf.placeholder_with_default(False, shape=(), name="training_bool")

X=tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y=tf.placeholder(tf.int64, shape=(None), name='y')


now=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
root_logdir="tf_logs"
logdir="{}/run-{}/".format(root_logdir, now)



with tf.name_scope('dnn'):

    hidden1=tf.layers.dense(X, n_hidden1, name='hidden1', activation=tf.nn.relu)

    hidden1_drop=tf.layers.dropout(hidden1, dropout_rate, training=training)

    hidden2=tf.layers.dense(hidden1_drop, n_hidden2, name='hidden2', activation=tf.nn.relu)

    hidden2_drop=tf.layers.dropout(hidden2, dropout_rate, training=training)

    #Output before softmax
    logits=tf.layers.dense(hidden2_drop, n_outputs, name='outputs')


with tf.name_scope('loss'):
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss=tf.reduce_mean(cross_entropy, name='loss')
    cross_ent_summary=tf.summary.scalar('cross_entropy', cross_entropy)



with tf.name_scope('train'):
    optimiser=tf.train.AdamOptimizer()
    training_op=optimiser.minimize(loss)

with tf.name_scope('eval'):
    correct=tf.nn.in_top_k(logits, y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct, tf.float32))
    acc_summary=tf.summary.scalar('Accuracy', accuracy)





#Summary
file_writer=tf.summary.FileWriter(logdir, tf.get_default_graph())


#Init and save
init = tf.global_variables_initializer()
saver=tf.train.Saver()




from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/")




with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples//batch_size):
            X_batch, y_batch=mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training:True})

            if iteration % 10 ==0:
                acc_summary_str=acc_summary.eval(feed_dict={X: X_batch, y: y_batch, training:True})
                step=epoch*(mnist.train.num_examples//batch_size)*iteration

                file_writer.add_summary(acc_summary_str, step)


        acc_train=accuracy.eval(feed_dict={X: X_batch, y: y_batch, training:True})

        acc_test=accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels, training:False})

        print "Epoch: {}. Train Accuracy: {}, Test Accuracy: {}".format(epoch, acc_train, acc_test)

    save_path = saver.save(sess, "./my_model.ckpt")


file_writer.close()