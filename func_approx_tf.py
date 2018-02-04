#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def target_func(x):
    return np.sin(x)


def inference(inputs):
    fc1 = tf.layers.dense(inputs=inputs, units=10, activation=tf.nn.relu, name="fc1")
    fc2 = tf.layers.dense(inputs=fc1, units=10, activation=tf.nn.relu, name="fc2")
    output = tf.layers.dense(inputs=fc2, units=1, activation=None, name="output")
    return output


def loss(truth, predict):
    losses = tf.reduce_sum(tf.square(truth-predict, name="loss"))
    return losses


def training(losses):
    return tf.train.AdamOptimizer().minimize(losses)


def main(argv=None):
    x = tf.placeholder(tf.float32, shape=(None, 1), name='inputs')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='truth')
    batch_size = 5
    predict = inference(x)

    losses = loss(y, predict)

    train_step = training(losses)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(100000):
            inp = (np.random.rand(batch_size, 1)-0.5)*10
            sess.run(train_step, feed_dict={x: inp, y: target_func(inp)})
            if i % 1000 == 0:
                inp = (np.random.rand(batch_size, 1)-0.5)*10
                loss_val = sess.run(losses, feed_dict={x: inp, y: target_func(inp)})
                print ('Step:%d, Loss:%f' % (i, loss_val))
            if i % 10000 == 0:
                rang = np.arange(-np.pi, np.pi, 0.1)
                rang2 = np.reshape(rang, (-1, 1))
                truth = target_func(rang)
                pred = sess.run(predict, feed_dict={x: rang2})
                plt.figure()
                plt.plot(rang, truth)
                plt.plot(rang, pred)
                plt.savefig('fig/'+'graph_'+str(i)+'.png')


if __name__ == '__main__':
    main()
