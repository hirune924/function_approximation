#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def target_func(x):
    term1 = np.sqrt(np.abs(x))
    term2 = np.sqrt(np.abs(6-np.square(x)))

    return term1+term2, term1-term2


def inference(inputs):
    fc1_1 = tf.layers.dense(inputs=inputs, units=100, activation=tf.nn.leaky_relu, name="fc1_1")
    fc2_1 = tf.layers.dense(inputs=fc1_1, units=100, activation=tf.nn.leaky_relu, name="fc2_1")
    output_1 = tf.layers.dense(inputs=fc2_1, units=1, activation=None, name="output_1")
    fc1_2 = tf.layers.dense(inputs=inputs, units=100, activation=tf.nn.leaky_relu, name="fc1_2")
    fc2_2 = tf.layers.dense(inputs=fc1_2, units=100, activation=tf.nn.leaky_relu, name="fc2_2")
    output_2 = tf.layers.dense(inputs=fc2_2, units=1, activation=None, name="output_2")
    return output_1, output_2


def loss(truth, predict):
    losses = tf.reduce_sum(tf.square(truth-predict, name="loss"))
    return losses


def training(losses):
    return tf.train.AdamOptimizer().minimize(losses)


def main(argv=None):
    x = tf.placeholder(tf.float32, shape=(None, 1), name='inputs')
    y = tf.placeholder(tf.float32, shape=(2, None, 1), name='truth')
    batch_size = 50
    predict = inference(x)

    losses = loss(y, predict)

    train_step = training(losses)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(100000):
            inp = (np.random.rand(batch_size, 1)-0.5)*2*np.sqrt(6)
            sess.run(train_step, feed_dict={x: inp, y: target_func(inp)})
            if i % 1000 == 0:
                inp = (np.random.rand(batch_size, 1)-0.5)*2*np.sqrt(6)
                loss_val = sess.run(losses, feed_dict={x: inp, y: target_func(inp)})
                print ('Step:%d, Loss:%f' % (i, loss_val))
            if i % 10000 == 0:
                rang = np.arange(-np.sqrt(6.05), np.sqrt(6.05), 0.00001)
                rang2 = np.reshape(rang, (-1, 1))
                truth1, truth2 = target_func(rang)
                pred1, pred2 = sess.run(predict, feed_dict={x: rang2})
                plt.figure()
                #plt.plot(rang, truth1)
                #plt.plot(rang, truth2)
                plt.plot(rang2, pred1)
                plt.plot(rang2, pred2)
                plt.savefig('fig/'+'graph_'+str(i)+'.png')


if __name__ == '__main__':
	main()


