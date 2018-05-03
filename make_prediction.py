import tensorflow as tf
import time
import numpy as np
import clean_data

FEATURE_SIZE = 40

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
swahili_parameters = ["result_sw.file", "docs_sw/", "labels_sw.txt", ".txt"]

#inputs, labels = get_data(*french_parameters)
inputs, labels = get_data(*english_parameters)
#inputs, labels = get_data(*swahili_parameters)
batch_shape = len(inputs)

X = tf.placeholder(tf.float32, shape=[batch_shape, FEATURE_SIZE], name='features')
Y = tf.placeholder(tf.float32, shape=[batch_shape, 20], name = 'labels')
model_hl = tf.placeholder(tf.int32, shape=(), name='model_hl')
model_lr = tf.placeholder(tf.float32, shape=(), name='model_lr')

W = tf.Variable(tf.truncated_normal([FEATURE_SIZE, model_hl]), name='W', validate_shape=False)
w = tf.Variable(tf.truncated_normal([model_hl, 20]), name='w', validate_shape=False)

c = tf.Variable(tf.zeros([model_hl]), name = 'c', validate_shape=False)
b = tf.Variable(tf.zeros([20]), name = 'b')

with tf.name_scope("hidden_layer") as scope:
    h = tf.nn.relu(tf.add(tf.matmul(X, W), c))

with tf.name_scope("output") as scope:
    y_estimated = tf.matmul(h, w) + b

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

with tf.Session() as sess:
    saver.restore(sess, "./en_weights/simple.weights")
    my_dict={X: np.array(inputs), Y: np.array(labels), model_lr: 0, model_hl: 20}
    preds, acc, batch_loss = sess.run([predictions, accuracy, loss], feed_dict=my_dict)
    print "Accuracy: {}, Predictions: {}, Actual Labels: {}".format(acc, preds, np.argmax(labels, axis=1))
