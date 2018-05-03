import tensorflow as tf
import time
import numpy as np
import clean_data

FEATURE_SIZE = 90
max_epochs = 10000
def get_data(filename, doc_loc, labels, doc_end):
    input_data = clean_data.cleaned_data(filename, doc_loc, labels, doc_end)
    inputs = []
    labels = []
    for key in input_data:
        inputs.append(input_data[key][0])
        labels.append(input_data[key][1])
    return inputs, labels

french_parameters = ["result.file", "docs/", "labels.txt", ".xml"]
english_parameters = ["result_en.file", "docs_en/", "labels_en.txt", ".translation.eng.txt"]
tagalog_parameters = ["result_tl.file", "docs_tl/", "labels_tl.txt", ".txt"]

#inputs, labels = get_data(*french_parameters)
inputs, labels = get_data(*english_parameters)
#inputs, labels = get_data(*tagalog_parameters)
batch_shape = len(inputs)

X = tf.placeholder(tf.float32, shape=[batch_shape, FEATURE_SIZE], name='features')
Y = tf.placeholder(tf.float32, shape=[batch_shape, 20], name = 'labels')
model_hl = tf.placeholder(tf.int32, shape=(), name='model_hl')
model_lr = tf.placeholder(tf.float32, shape=(), name='model_lr')

W = tf.Variable(tf.truncated_normal([FEATURE_SIZE, model_hl]), name='W', validate_shape=False)
w = tf.Variable(tf.truncated_normal([10, 20]), name='w', validate_shape=False)
W2 = tf.Variable(tf.truncated_normal([model_hl, 10]), name='W2', validate_shape=False)

c = tf.Variable(tf.zeros([model_hl]), name = 'c', validate_shape=False)
b2 = tf.Variable(tf.zeros([10]), name = 'b2', validate_shape=False)
b = tf.Variable(tf.zeros([20]), name = 'b')

with tf.name_scope("h1") as scope:
    h = tf.nn.relu(tf.add(tf.matmul(X, W), c))

with tf.name_scope("h2") as scope:
    h2 = tf.nn.sigmoid(tf.matmul(h, W2) + b2)

with tf.name_scope("output") as scope:
    y_estimated = tf.matmul(h2, w) + b

with tf.name_scope("loss") as scope:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_estimated, labels=Y))

with tf.name_scope("train") as scope:
    opt = tf.train.AdamOptimizer(model_lr)
    train_step = opt.minimize(loss)

with tf.name_scope('test') as scope:
    predictions = tf.argmax(y_estimated, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, tf.argmax(labels, axis=1)), tf.float32))
    ## [nb_queries]


init = tf.global_variables_initializer()
saver = tf.train.Saver()

best_val_loss = float('inf')
for hl in xrange(5, 30, 5):
    numbers = [.0005, .0006, .0007, .0008, .0009, .001, .002, .003, .004, .005]
    for lr in numbers:
        print lr
        with tf.Session() as sess:
            my_dict={X: np.array(inputs), Y: np.array(labels), model_lr: lr, model_hl: hl}
            sess.run(init, feed_dict=my_dict)
            for epoch in range(max_epochs):
                _, preds, acc, batch_loss = sess.run([train_step, predictions, accuracy, loss], feed_dict=my_dict)
                if epoch % 500 == 0:
                    print "_"*80
                    print "Epoch: ", epoch
                    '''print "    y_estimated: "
                    for element in sess.run(y_estimated, feed_dict={X: inputs, Y: labels}):
                        print "     ", element
                    print "    w:  "
                    for element in sess.run(w):
                        print "    ", element
                    print "    b:  "
                    for element in sess.run(b):
                        print "    ", element
                    print "    W:  "
                    for element in sess.run(W):
                        print "    ", element
                    print "    c:  "
                    for element in sess.run(c):
                        print "    ", element'''
                    print "    loss: {} acc {} preds{} labels{}".format(batch_loss, acc, preds, tf.argmax(labels, axis=1).eval())
                    with open("./loss_graph_en_complex/"+str(hl)+"_"+str(lr)+".txt", "a+") as fp:
                        fp.write(str(epoch) + "," + str(batch_loss))
                        fp.write("\n")
                if batch_loss < best_val_loss:
                    best_val_loss = batch_loss
                    saver.save(sess, './en_weights/complex.weights')
            with open("complex_results/en_results.txt", "a") as fp:
                fp.write("Hidden Layer Size: {}, Learning Rate: {}, Loss: {}, Acc: {}".format(hl, lr, batch_loss, acc))
                fp.write('\n')
