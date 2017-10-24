from importData import Dataset
import inference
training = Dataset('wxb_pic/pic', '.jpg')
testing = Dataset('wxb_pic/pic_test', '.jpg')

import tensorflow as tf

# Parameters
learn_rate = 0.001 # 事先设定的初始学习率
decay_rate = 0.1 # 衰减系数
batch_size = 64
display_step = 20

n_classes = training.num_labels # we got mad kanji
dropout = 0.8 # Dropout, probability to keep units
imagesize = 227
img_channel = 3

x = tf.placeholder(tf.float32, [None, imagesize, imagesize, img_channel])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

pred = inference.alex_net(x, keep_prob, n_classes, imagesize, img_channel)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
 
# tf.train.exponential_decay函数实现指数衰减学习率 1.首先使用较大学习率(目的：为快速得到一个比较优的解);2.然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定);
# learning_rate：0.001；staircase=True;则每1000轮训练后要乘以decay_rate.
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learn_rate, global_step, 1000, decay_rate, staircase=True) # 生成学习率
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, global_step=global_step)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
saver = tf.train.Saver() # Saver类提供了向checkpoints文件保存和从checkpoints文件中恢复变量的相关方法。
                         # Checkpoints文件是一个二进制文件，它把变量名映射到对应的tensor值 。
tf.add_to_collection("x", x)# tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表
tf.add_to_collection("y", y)
tf.add_to_collection("keep_prob", keep_prob)
tf.add_to_collection("pred", pred)
tf.add_to_collection("accuracy", accuracy)

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step < 3000: # 训练次数
        batch_ys, batch_xs = training.nextBatch(batch_size)# 每训练一次随机抓取训练数据中的batch_size 64个批处理数据点
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0: # 每20次，记录一次准确率、损失、学习率
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            rate = sess.run(lr)
            print( "lr" + str(rate) + " Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

        if step % 1000 == 0:# 每训练1000次保存一下checkpoints
            saver.save(sess, 'save/model.ckpt', global_step=step*batch_size)
            # 使用Saver.save()方法保存模型：sess：表示当前会话，当前会话记录了当前的变量值
                                         # checkpoint_dir + 'model.ckpt'：表示存储的文件名
                                         # global_step：表示当前是第几步
        step += 1
    print( "Optimization Finished!")
    step_test = 1
    while step_test * batch_size < len(testing):
        testing_ys, testing_xs = testing.nextBatch(batch_size)
         print( "Testing Accuracy:", sess.run(accuracy, feed_dict={x: testing_xs, y: testing_ys, keep_prob: 1.}))
        step_test += 1
